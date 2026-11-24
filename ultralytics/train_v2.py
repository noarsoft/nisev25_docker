# train_v2.py
# Batch YOLO training driven by a json config, with a manifest per run (resume / redo) and per-epoch timing.
#
#   python train_v2.py                              train every fold of the groups selected in train_config.json
#   python train_v2.py --status                     show which folds are done / failed / need redo (no training)
#   python train_v2.py --dry-run                    show the plan only
#   python train_v2.py --only "iris_nopca_*"        only groups whose name matches (glob); group = <dataset>_<mode>_algo<a>_k<k>
#   python train_v2.py --redo                       retrain the selected folds even if done (their old results are deleted)
#   python train_v2.py --clear                      delete results + manifest of the selected groups (asks first; --yes, --dry-run)
#   python train_v2.py --config other.json
#
# Any value of the config can be overridden on the command line (command line wins):
#   --datasets iris,wine  --modes nopca  --algos 1,2  --k_list 8  --folds 0,1
#   --weights yolov8m.pt  --epochs 300  --imgsz 640  --batch 32  --workers 4  --device 1  --seed 7  --patience 50
#   --extra fliplr=0.0 mosaic=0      (more ultralytics arguments; key=value, or one json object)  --data-root out  --project-root runs_strip
# The merged (effective) config is saved as train_config.used.json next to the manifest.
#
# Layout:  <project_root>/<group>/train_manifest.json        status of every fold, metrics, timing, versions
#          <project_root>/<group>/train_config.used.json     the config this group was run with
#          <project_root>/<group>/fold_<f>/train/weights/best.pt , results.csv, ...
#          <project_root>/<group>/fold_<f>/epochs.csv         seconds per epoch (epoch, epoch_sec, elapsed_sec)
#
# Rules: a fold already "done" with the same settings is skipped; "running" (crashed/killed), "failed" and
# "interrupted" folds are cleared and trained again; a "done" fold whose settings changed is reported as STALE
# and kept until you pass --redo. Different groups can run in parallel (one process per GPU, different --only).

import argparse
import csv
import fnmatch
import hashlib
import itertools
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "train_config.json"
MANIFEST = "train_manifest.json"
NOT_IN_FINGERPRINT = ("device", "workers")  # change speed, not what is trained


def now():
    return datetime.now().isoformat(timespec="seconds")


def resolve(p):
    p = Path(p).expanduser()
    return p if p.is_absolute() else HERE / p


def load_config(path):
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        sys.exit(f"config not found: {path}")
    except json.JSONDecodeError as e:
        sys.exit(f"{path} is not valid json: {e}")
    for key in ("select", "train"):
        if key not in cfg:
            sys.exit(f"{path} is missing key: {key}")
    cfg.setdefault("data_root", "out")
    cfg.setdefault("project_root", "runs_strip")
    sel = cfg["select"]
    for key in ("datasets", "modes", "algos", "k_list"):
        if key not in sel:
            sys.exit(f"{path}: select.{key} is missing")
    sel.setdefault("folds", [0, 1, 2, 3, 4])
    for key in ("weights", "epochs", "imgsz", "batch", "device"):
        if key not in cfg["train"]:
            sys.exit(f"{path}: train.{key} is missing")
    return cfg


def expand_groups(cfg, only):
    data_root = resolve(cfg["data_root"])
    sel = cfg["select"]
    datasets = sel["datasets"]
    if datasets in ("*", ["*"]):
        datasets = sorted(p.name for p in data_root.iterdir() if p.is_dir() and p.name != "split") if data_root.is_dir() else []
    groups = []
    for d, m, a, k in itertools.product(datasets, sel["modes"], sel["algos"], sel["k_list"]):
        name = f"{d}_{m}_algo{a}_k{k}"
        if only and not fnmatch.fnmatch(name, only):
            continue
        groups.append({"group": name, "dataset": d, "mode": m, "algo": int(a), "k": int(k),
                       "base_dir": data_root / d / m / f"algo{a}" / f"repacking_k_{k}"})
    return groups


def split_id_of(base_dir):
    try:
        with open(base_dir.parent / "manifest.json", "r", encoding="utf-8-sig") as f:
            return json.load(f).get("split_id")
    except (OSError, json.JSONDecodeError):
        return None


def fingerprint(train_cfg, split_id):
    t = {k: v for k, v in train_cfg.items() if k not in NOT_IN_FINGERPRINT}
    return hashlib.sha256(json.dumps({"train": t, "split_id": split_id}, sort_keys=True).encode()).hexdigest()[:16]


def load_manifest(gdir):
    p = gdir / MANIFEST
    if p.is_file():
        with open(p, "r", encoding="utf-8-sig") as f:  # tolerate a BOM added by Windows editors/PowerShell
            return json.load(f)
    return None


def save_manifest(gdir, manifest):
    gdir.mkdir(parents=True, exist_ok=True)
    tmp = gdir / (MANIFEST + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    os.replace(tmp, gdir / MANIFEST)  # never leaves a half-written manifest


def versions():
    v = {"python": sys.version.split()[0]}
    try:
        import torch
        import ultralytics
        v.update({"torch": torch.__version__, "ultralytics": ultralytics.__version__})
    except Exception:
        pass
    return v


def gpu_name(device):
    try:
        import torch
        first = str(device).split(",")[0].strip()
        if first.isdigit() and torch.cuda.is_available():
            return torch.cuda.get_device_name(int(first))
    except Exception:
        pass
    return None


def make_trainer(overrides):
    from ultralytics.models.yolo.detect import DetectionTrainer
    return DetectionTrainer(overrides=overrides)


def decide(entry, fp, fold_dir, redo):
    """-> (action, reason); action is run / skip / stale"""
    if redo:
        return "run", "--redo"
    if entry is None:
        return "run", "not trained yet"
    st = entry.get("status")
    if st == "done":
        if not (fold_dir / "train" / "weights" / "best.pt").is_file():
            return "run", "done but best.pt is missing"
        if entry.get("fingerprint") != fp:
            return "stale", "settings or data changed since it was trained (use --redo to retrain)"
        return "skip", "done"
    return "run", f"previous status: {st}"


def run_fold(group, fold, train_cfg, gdir, fp, manifest):
    fold_dir = gdir / f"fold_{fold}"
    yaml = group["base_dir"] / str(fold) / "dataset.yaml"
    entry = {"status": "running", "started_at": now(), "fingerprint": fp, "finished_at": None, "error": None}
    manifest["folds"][str(fold)] = entry
    if not yaml.is_file():
        entry.update(status="failed", finished_at=now(), error=f"data not found: {yaml}")
        save_manifest(gdir, manifest)
        return entry

    if fold_dir.exists():
        shutil.rmtree(fold_dir)  # fresh start: old weights/results of this fold are removed
    fold_dir.mkdir(parents=True)
    entry["device_name"] = gpu_name(train_cfg["device"])
    save_manifest(gdir, manifest)

    overrides = {
        "model": train_cfg["weights"], "data": str(yaml), "epochs": train_cfg["epochs"], "imgsz": train_cfg["imgsz"],
        "batch": train_cfg["batch"], "workers": train_cfg.get("workers", 1), "device": train_cfg["device"],
        "seed": train_cfg.get("seed", 0), "patience": train_cfg.get("patience", 100),
        "project": str(fold_dir), "name": "train", "exist_ok": True,
    }
    overrides.update(train_cfg.get("extra", {}))

    epochs_csv = fold_dir / "epochs.csv"
    epoch_secs = []
    seen = set()
    t0 = time.time()

    def on_epoch_end(trainer):
        if trainer.epoch in seen:  # ultralytics fires this callback again for the final validation of best.pt
            return
        seen.add(trainer.epoch)
        sec = float(trainer.epoch_time or 0.0)
        epoch_secs.append(sec)
        new = not epochs_csv.exists()
        with open(epochs_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["epoch", "epoch_sec", "elapsed_sec"])
            w.writerow([trainer.epoch + 1, round(sec, 3), round(time.time() - t0, 3)])

    try:
        trainer = make_trainer(overrides)
        trainer.add_callback("on_fit_epoch_end", on_epoch_end)
        trainer.train()
        metrics = {k: float(v) for k, v in (getattr(trainer, "metrics", None) or {}).items() if str(k).startswith("metrics/")}
        stopper = getattr(trainer, "stopper", None)
        entry.update(
            status="done", finished_at=now(), elapsed_sec=round(time.time() - t0, 2),
            epochs_run=len(epoch_secs) or int(trainer.epoch) + 1, best_epoch=getattr(stopper, "best_epoch", None),
            sec_per_epoch=round(sum(epoch_secs) / len(epoch_secs), 3) if epoch_secs else None,
            metrics=metrics, best_pt=str(Path(trainer.best).relative_to(gdir)) if getattr(trainer, "best", None) else None,
        )
    except KeyboardInterrupt:
        entry.update(status="interrupted", finished_at=now(), elapsed_sec=round(time.time() - t0, 2))
        save_manifest(gdir, manifest)
        raise
    except Exception as e:  # keep going with the next fold; the error is in the manifest
        entry.update(status="failed", finished_at=now(), elapsed_sec=round(time.time() - t0, 2),
                     error=f"{type(e).__name__}: {str(e)[:400]}")
    save_manifest(gdir, manifest)
    return entry


def _ts(text):
    try:
        return datetime.fromisoformat(text) if text else None
    except ValueError:
        return None


def group_times(folds):
    """(start, stop, interval text): start = first fold start, stop = last fold finish (only when every fold is done),
    interval = stop - start (wall clock, includes waiting between folds); while unfinished it runs to now and ends with '*'"""
    starts = [t for t in (_ts(x.get("started_at")) for x in folds) if t]
    if not starts:
        return "-", "-", "-"
    start = min(starts)
    ends = [t for t in (_ts(x.get("finished_at")) for x in folds) if t]
    complete = all(x.get("status") == "done" for x in folds) and ends
    end = max(ends) if complete else datetime.now()
    secs = max(0, int((end - start).total_seconds()))
    span = f"{secs // 3600}h{secs % 3600 // 60:02d}m" + ("" if complete else "*")
    return start.strftime("%m-%d %H:%M"), (end.strftime("%m-%d %H:%M") if complete else "-"), span


def cmd_status(cfg):
    root = resolve(cfg["project_root"])
    rows = sorted(root.glob(f"*/{MANIFEST}")) if root.is_dir() else []
    if not rows:
        print(f"no manifests under {root}")
        return
    print(f"{'group':36} {'done':>5} {'failed':>6} {'other':>5} {'mAP50-95':>9} {'sec/epoch':>10} {'train min':>9} "
          f"{'start':>11} {'stop':>11} {'interval':>9}")
    for p in rows:
        with open(p, "r", encoding="utf-8-sig") as f:
            m = json.load(f)
        folds = m["folds"].values()
        done = [x for x in folds if x.get("status") == "done"]
        failed = [x for x in folds if x.get("status") == "failed"]
        maps = [x["metrics"].get("metrics/mAP50-95(B)") for x in done if x.get("metrics")]
        maps = [v for v in maps if v is not None]
        spe = [x["sec_per_epoch"] for x in done if x.get("sec_per_epoch")]
        tot = sum(x.get("elapsed_sec") or 0 for x in done) / 60
        t_start, t_stop, t_span = group_times(list(folds))
        print(f"{m['group']:36} {len(done):>5} {len(failed):>6} {len(m['folds']) - len(done) - len(failed):>5} "
              f"{(sum(maps) / len(maps) if maps else float('nan')):>9.4f} "
              f"{(sum(spe) / len(spe) if spe else float('nan')):>10.2f} {tot:>9.1f} "
              f"{t_start:>11} {t_stop:>11} {t_span:>9}")
        for k, x in sorted(m["folds"].items()):
            if x.get("status") != "done":
                print(f"    fold {k}: {x.get('status')}  {x.get('error') or ''}")


def csv_list(text, cast=str):
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def apply_overrides(cfg, a):
    """command line > config file: only options that were actually given replace the config values"""
    sel, train = cfg["select"], cfg["train"]
    if a.data_root is not None:
        cfg["data_root"] = a.data_root
    if a.project_root is not None:
        cfg["project_root"] = a.project_root
    if a.datasets is not None:
        sel["datasets"] = "*" if a.datasets.strip() == "*" else csv_list(a.datasets)
    if a.modes is not None:
        sel["modes"] = csv_list(a.modes)
    if a.algos is not None:
        sel["algos"] = csv_list(a.algos, int)
    if a.k_list is not None:
        sel["k_list"] = csv_list(a.k_list, int)
    if a.folds is not None:
        sel["folds"] = csv_list(a.folds, int)
    for key in ("weights", "epochs", "imgsz", "batch", "workers", "device", "seed", "patience"):
        val = getattr(a, key)
        if val is not None:
            train[key] = val
    if a.extra:
        extra = {}
        for item in a.extra:
            if item.lstrip().startswith("{"):  # a whole json object
                try:
                    extra.update(json.loads(item))
                except json.JSONDecodeError as e:
                    sys.exit(f"--extra {item!r} is not valid json (on Windows shells prefer key=value): {e}")
            elif "=" in item:
                key, _, val = item.partition("=")
                try:
                    extra[key.strip()] = json.loads(val)  # numbers, true/false, null
                except json.JSONDecodeError:
                    extra[key.strip()] = val
            else:
                sys.exit(f"--extra expects key=value or a json object, got {item!r}")
        train["extra"] = {**train.get("extra", {}), **extra}
    return cfg


def cmd_clear(cfg, a):
    """delete the run folder (results + manifest) of the selected groups"""
    root = resolve(cfg["project_root"])
    if cfg["select"]["datasets"] in ("*", ["*"]):  # data may already be moved away: look at the runs instead
        names = sorted(p.parent.name for p in root.glob(f"*/{MANIFEST}")) if root.is_dir() else []
        names = [n for n in names if not a.only or fnmatch.fnmatch(n, a.only)]
    else:
        names = [g["group"] for g in expand_groups(cfg, a.only)]
    targets = [root / n for n in names
               if (root / n / MANIFEST).is_file() or any((root / n).glob("fold_*"))]
    if not targets:
        print(f"nothing to clear under {root}")
        return
    print("will delete (weights, results and manifest):")
    for t in targets:
        print(f"  - {t}  ({len(list(t.glob('fold_*')))} fold folder(s))")
    if a.dry_run:
        print("(dry run, nothing deleted)")
        return
    if not a.yes and input("continue? [y/N] ").strip().lower() != "y":
        print("cancelled")
        return
    for t in targets:
        shutil.rmtree(t)
    print(f"deleted {len(targets)} group folder(s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--only", default=None, help="glob on the group name, e.g. 'iris_nopca_*'")
    ap.add_argument("--redo", action="store_true", help="retrain the selected folds even if done")
    ap.add_argument("--clear", action="store_true", help="delete results + manifest of the selected groups, then stop")
    ap.add_argument("--yes", "-y", action="store_true", help="--clear: no confirmation")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    o = ap.add_argument_group("overrides (command line wins over the config file)")
    o.add_argument("--data-root", dest="data_root", default=None)
    o.add_argument("--project-root", dest="project_root", default=None)
    o.add_argument("--datasets", default=None, help="comma-separated or '*'")
    o.add_argument("--modes", default=None, help="e.g. nopca,pca")
    o.add_argument("--algos", default=None, help="e.g. 1,2")
    o.add_argument("--k_list", default=None, help="e.g. 8 or 0,2,4,8")
    o.add_argument("--folds", default=None, help="e.g. 0,1")
    o.add_argument("--weights", default=None)
    o.add_argument("--epochs", type=int, default=None)
    o.add_argument("--imgsz", type=int, default=None)
    o.add_argument("--batch", type=int, default=None)
    o.add_argument("--workers", type=int, default=None)
    o.add_argument("--device", default=None)
    o.add_argument("--seed", type=int, default=None)
    o.add_argument("--patience", type=int, default=None)
    o.add_argument("--extra", nargs="+", default=None, metavar="KEY=VALUE",
                   help="more ultralytics arguments, merged into train.extra, e.g. --extra fliplr=0.0 mosaic=0 (or one json object)")
    a = ap.parse_args()

    cfg = apply_overrides(load_config(a.config), a)
    if a.status:
        cmd_status(cfg)
        return
    if a.clear:
        cmd_clear(cfg, a)
        return
    train_cfg = cfg["train"]
    folds = [int(x) for x in cfg["select"]["folds"]]
    groups = expand_groups(cfg, a.only)
    if not groups:
        sys.exit("no group selected (check select.* in the config and --only)")
    project_root = resolve(cfg["project_root"])
    counts = {"trained": 0, "skipped": 0, "stale": 0, "failed": 0, "missing": 0}

    if not a.dry_run:
        try:
            import ultralytics  # noqa: F401  fail fast instead of marking every fold as failed
        except ImportError:
            sys.exit("ultralytics is not installed for this python. Activate the venv first "
                     "(e.g. source ~/venv-yolo/bin/activate) or: pip install ultralytics")

    try:
        for g in groups:
            gdir = project_root / g["group"]
            if not g["base_dir"].is_dir():
                print(f"[{g['group']}] MISSING data: {g['base_dir']}  (export it: python transfer.py export --what sets ...)")
                counts["missing"] += 1
                continue
            sid = split_id_of(g["base_dir"])
            fp = fingerprint(train_cfg, sid)
            manifest = load_manifest(gdir) or {"group": g["group"], "dataset": g["dataset"], "mode": g["mode"],
                                               "algo": g["algo"], "k": g["k"], "folds": {}}
            manifest.update(base_dir=str(g["base_dir"]), split_id=sid, config=train_cfg, versions=versions())
            plan = []
            for f in folds:
                action, why = decide(manifest["folds"].get(str(f)), fp, gdir / f"fold_{f}", a.redo)
                plan.append((f, action, why))
            print(f"[{g['group']}] " + ", ".join(f"fold {f}: {act}" + (f" ({why})" if act != "skip" else "") for f, act, why in plan))
            if a.dry_run:
                continue
            if any(act == "run" for _, act, _ in plan):
                gdir.mkdir(parents=True, exist_ok=True)
                with open(gdir / "train_config.used.json", "w", encoding="utf-8") as dst:  # effective config: file + command line overrides
                    json.dump({**cfg, "cli": sys.argv[1:]}, dst, indent=2, ensure_ascii=False)
            for f, act, why in plan:
                if act == "skip":
                    counts["skipped"] += 1
                elif act == "stale":
                    counts["stale"] += 1
                else:
                    print(f"[{g['group']}] training fold {f} ...")
                    e = run_fold(g, f, train_cfg, gdir, fp, manifest)
                    if e["status"] == "done":
                        counts["trained"] += 1
                        print(f"[{g['group']}] fold {f} done in {e['elapsed_sec']}s, {e['epochs_run']} epochs, "
                              f"{e['sec_per_epoch']} s/epoch")
                    else:
                        counts["failed"] += 1
                        print(f"[{g['group']}] fold {f} FAILED: {e['error']}")
            if any(act == "run" for _, act, _ in plan):
                save_manifest(gdir, manifest)
    except KeyboardInterrupt:
        print("\ninterrupted: the running fold is marked 'interrupted' and will be retrained next time")
        sys.exit(130)

    print("summary:", ", ".join(f"{k}={v}" for k, v in counts.items()))
    sys.exit(1 if counts["failed"] or counts["missing"] else 0)


if __name__ == "__main__":
    main()
