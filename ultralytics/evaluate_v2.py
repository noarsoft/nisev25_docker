# evaluate_v2.py
# Test-set evaluation of every trained fold: one command instead of predict_v2.py --test per fold, with the mean over the folds.
#
#   python evaluate_v2.py --group iris_nopca_algo1_k8             all trained folds of one group
#   python evaluate_v2.py                                         every group that has a train_manifest.json under project_root
#   python evaluate_v2.py --only "iris_*" --folds 0,1,2           a subset (glob on the group name / list of folds)
#   python evaluate_v2.py --device 0 --conf 0.001
#
# Per fold: accuracy, balanced accuracy (mean recall per class), macro F1 and the number of test images with no detection
# (counted as wrong). Per group: mean and standard deviation over the folds, accuracy pooled over all test images,
# and the confusion matrix summed over the folds. The class of an image = its most confident detected box (as predict_v2.py).
# Needs, per fold: best.pt (train_v2.py), the algo manifest.json and the split test csv under data_root
#   python transfer.py export --what sets --datasets <ds> --modes <mode> --algos <a> --k_list <k>
#   python transfer.py export --what folder --path <ds>/split
# Writes predictions/eval_<group>.json and predictions/eval_summary.csv (one row per group and fold).
# The folds are a cross-validation of one dataset: the mean over folds is the result to report; best.pt of every fold was
# picked on that fold's validation set, the test set was not used for it.

import argparse
import csv
import fnmatch
import json
import statistics
import sys
from pathlib import Path

import numpy as np

import predict_v2 as pv
import train_v2

HERE = Path(__file__).resolve().parent


def fold_metrics(cm):
    """cm: K x (K+1) (rows = true class, columns = predicted class, last column = no detection)"""
    k = cm.shape[0]
    n = int(cm.sum())
    support = cm.sum(axis=1)
    pred_n = cm[:, :k].sum(axis=0)
    rec = [cm[c, c] / support[c] for c in range(k) if support[c] > 0]
    f1 = []
    for c in range(k):
        if support[c] == 0:
            continue
        p = cm[c, c] / pred_n[c] if pred_n[c] else 0.0
        r = cm[c, c] / support[c]
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
    return {"n": n, "correct": int(np.trace(cm[:, :k])), "accuracy": float(np.trace(cm[:, :k]) / n) if n else float("nan"),
            "balanced_accuracy": float(np.mean(rec)) if rec else float("nan"), "macro_f1": float(np.mean(f1)) if f1 else float("nan"),
            "no_detection": int(cm[:, k].sum())}


def mean_std(vals):
    vals = [v for v in vals if v == v]
    if not vals:
        return float("nan"), float("nan")
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def evaluate_fold(a, group, fold, YOLO):
    ctx = pv.load_context(group, fold, a.config, a.data_root, a.project_root, None)
    csv_path = ctx.data_root / ctx.ds / "split" / ctx.mode / f"fold_{fold}" / "test.csv"
    if not csv_path.is_file():
        sys.exit(f"{csv_path} not found (python transfer.py export --what folder --path {ctx.ds}/split)")
    items, _ = pv.rows_to_items(ctx, pv.read_csv(csv_path), "test", ctx.label_col, True, str(csv_path))
    if not any(t is not None and t >= 0 for _, _, t, _ in items):
        sys.exit(f"no usable true-class column in {csv_path} (label column: {ctx.label_col})")
    weights = pv.resolve_weights(ctx)
    model = YOLO(str(weights))
    kw = {"imgsz": ctx.canvas, "conf": a.conf, "verbose": False}
    if a.device is not None:
        kw["device"] = a.device
    k = len(ctx.labels)
    cm = np.zeros((k, k + 1), dtype=int)
    for s in range(0, len(items), a.batch):
        chunk = items[s:s + a.batch]
        for (_, _, truth, _), r in zip(chunk, model.predict([im for _, im, _, _ in chunk], **kw)):
            if truth is None or truth < 0:
                continue
            sc = pv.class_scores(r, k)
            cm[truth, int(np.argmax(sc)) if max(sc) > 0 else k] += 1
    return ctx.labels, cm, str(weights)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", default=None, help="one group; default: every group with a manifest (see --only)")
    ap.add_argument("--only", default=None, help="glob on the group name")
    ap.add_argument("--folds", default=None, help="comma-separated; default: every fold that is done")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out-dir", default=str(HERE / "predictions"))
    ap.add_argument("--config", default=str(train_v2.DEFAULT_CONFIG))
    ap.add_argument("--data-root", dest="data_root", default=None)
    ap.add_argument("--project-root", dest="project_root", default=None)
    a = ap.parse_args()

    cfg = train_v2.load_config(a.config)
    project_root = train_v2.resolve(a.project_root or cfg["project_root"])
    if a.group:
        groups = [a.group]
    else:
        groups = sorted(p.parent.name for p in project_root.glob(f"*/{train_v2.MANIFEST}"))
        groups = [g for g in groups if not a.only or fnmatch.fnmatch(g, a.only)]
    if not groups:
        sys.exit(f"no trained group found under {project_root}")
    want = {int(x) for x in a.folds.split(",")} if a.folds else None

    from ultralytics import YOLO
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_out, problems = [], 0

    for group in groups:
        tm = pv.load_json(project_root / group / train_v2.MANIFEST)
        if not tm:
            print(f"[{group}] no training manifest, skipped")
            problems += 1
            continue
        folds = sorted(int(f) for f, e in tm["folds"].items() if e.get("status") == "done" and (want is None or int(f) in want))
        if not folds:
            print(f"[{group}] no finished fold to evaluate")
            problems += 1
            continue
        print(f"\n[{group}]")
        per_fold, total, labels = [], None, None
        for f in folds:
            try:
                labels, cm, weights = evaluate_fold(a, group, f, YOLO)
            except SystemExit as e:  # a missing file of one fold must not stop the others
                print(f"  fold {f}: skipped: {e.code}")
                problems += 1
                continue
            m = fold_metrics(cm)
            m["fold"] = f
            per_fold.append(m)
            total = cm if total is None else total + cm
            rows_out.append({"group": group, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}})
        if not per_fold:
            continue

        print(f"  {'fold':>4} {'n':>5} {'accuracy':>9} {'balanced':>9} {'macro F1':>9} {'no det.':>8}")
        for m in per_fold:
            print(f"  {m['fold']:>4} {m['n']:>5} {m['accuracy']:>9.4f} {m['balanced_accuracy']:>9.4f} {m['macro_f1']:>9.4f} {m['no_detection']:>8}")
        summary = {"group": group, "folds": [m["fold"] for m in per_fold], "per_fold": per_fold, "labels": labels}
        for key in ("accuracy", "balanced_accuracy", "macro_f1"):
            mu, sd = mean_std([m[key] for m in per_fold])
            summary[key] = {"mean": mu, "std": sd}
        pooled = sum(m["correct"] for m in per_fold) / sum(m["n"] for m in per_fold)
        summary["pooled_accuracy"] = pooled
        summary["confusion_matrix_sum"] = {"rows": "true", "columns": labels + ["no detection"], "matrix": total.tolist()}
        a_ = summary["accuracy"]
        print(f"  mean over {len(per_fold)} fold(s): accuracy {a_['mean']:.4f} +/- {a_['std']:.4f}   "
              f"balanced {summary['balanced_accuracy']['mean']:.4f}   macro F1 {summary['macro_f1']['mean']:.4f}   "
              f"(pooled accuracy {pooled:.4f}; +/- = std over folds)")
        w0 = max(len(c) for c in labels) + 1
        print("  confusion matrix summed over the folds (rows = true, columns = predicted):")
        print("  " + " " * w0 + "".join(f"{c[:8]:>9}" for c in labels) + f"{'none':>9}")
        for i, c in enumerate(labels):
            print("  " + f"{c:<{w0}}" + "".join(f"{total[i, j]:>9}" for j in range(len(labels) + 1)))
        with open(out_dir / f"eval_{group}.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)

    if rows_out:
        cols = ["group", "fold", "n", "correct", "accuracy", "balanced_accuracy", "macro_f1", "no_detection"]
        with open(out_dir / "eval_summary.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows({c: r[c] for c in cols} for r in rows_out)
        print(f"\nwritten: {out_dir / 'eval_summary.csv'} and eval_<group>.json")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
