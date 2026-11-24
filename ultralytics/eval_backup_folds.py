
# ไม่ต้องใช้แล้ว ใช้แค่ eval_backup_folds.py


# python eval_backup_folds.py --data_base "C:\yolo8_strip\ultralytics\backup\breast_cancer\output_k_fold_boost_nopca" --weights_base "C:\yolo8_strip\ultralytics\backup\breast_cancer\plain_output_k_fold_boost_nopca" --folds 0 1 2 3 4 --device 0 --conf 0.01 --imgsz 640

# eval_folds_test2.py
# Evaluate YOLO weights on test set across folds + save per-image CSV + timing (ms/image)
#
# Examples:
#   # old structure (base_dir has fold/test + runs/foldX/train*/weights)
#   python eval_folds_test2.py --data_base "C:\yolo8_strip\ultralytics\output_k_fold_boost_nopca" --folds 0 1 2 3 4 --device 0 --mode auto
#
#   # backup structure (data_base has fold/test, weights_base has plain_output_k_fold.../fold/train/weights)
#   python eval_folds_test2.py ^
#     --data_base "C:\yolo8_strip\ultralytics\backup\breast_cancer\output_k_fold_boost_nopca" ^
#     --weights_base "C:\yolo8_strip\ultralytics\backup\breast_cancer\plain_output_k_fold_boost_nopca" ^
#     --folds 0 1 2 3 4 --device 0 --mode plain
#
# Notes:
# - Supports multi-class automatically (reads classes from dataset.yaml if present).
# - Saves per-image: y_true, y_pred, missed, and timing (preprocess/inference/postprocess/total).
# - Timing:
#     - res[0].speed gives preprocess/inference/postprocess (ms)
#     - total_ms is measured by perf_counter around model.predict()

import os
import time
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml  # pip install pyyaml

from ultralytics import YOLO
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def collect_images(root: Path):
    if not root.exists():
        return []
    return [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS and p.is_file()]


def img_to_gt_label_path(img_path: Path) -> Path:
    parts = list(img_path.parts)
    parts_lower = [p.lower() for p in parts]
    if "images" in parts_lower:
        i = parts_lower.index("images")
        parts[i] = "labels"
        return Path(*parts).with_suffix(".txt")
    return img_path.with_suffix(".txt")


def read_first_gt_class(label_path: Path):
    if not label_path.exists():
        return None
    lines = [ln.strip() for ln in label_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return None
    try:
        return int(lines[0].split()[0])
    except Exception:
        return None


def pick_image_label_from_detections(result):
    """Pick class of the highest-confidence detection. If no detection -> -1."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return -1
    confs = boxes.conf.cpu().tolist()
    clss = boxes.cls.cpu().tolist()
    k = int(max(range(len(confs)), key=lambda i: confs[i]))
    return int(clss[k])


def pick_image_label_weighted_vote(result, conf_vote_min=0.25, allowed_classes=None):
    """Optional: weighted voting by sum of confidences per class."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return -1

    confs = boxes.conf.cpu().tolist()
    clss = boxes.cls.cpu().tolist()

    score = {}  # class_id -> sum(conf)
    for c, conf in zip(clss, confs):
        c = int(c)
        if allowed_classes is not None and c not in allowed_classes:
            continue
        if float(conf) < conf_vote_min:
            continue
        score[c] = score.get(c, 0.0) + float(conf)

    if not score:
        return -1
    return max(score.items(), key=lambda kv: kv[1])[0]


def read_classes_from_dataset_yaml(dataset_yaml: Path):
    """
    Returns (nc, class_ids_list) if dataset.yaml exists.
    If not found, return (None, None) and we'll infer from y_true later.
    """
    if not dataset_yaml.exists():
        return None, None
    try:
        y = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
        nc = int(y.get("nc")) if "nc" in y else None
        if nc is not None:
            return nc, list(range(nc))
    except Exception:
        pass
    return None, None


def find_weights_for_fold(
    fold,
    mode,
    data_base,
    weights_base,
):
    """
    Try multiple layouts (auto):
      A) old pipeline: data_base/runs/fold{fold}/train_*/weights/best.pt
      B) plain layout: data_base/{fold}/train/weights/best.pt
      C) backup plain layout: weights_base/{fold}/train/weights/best.pt
    """
    # A) old pipeline layout
    runs_fold = data_base / "runs" / f"fold{fold}"
    w_hard = runs_fold / "train_hard" / "weights" / "best.pt"
    w_norm = runs_fold / "train1" / "weights" / "best.pt"
    w_plain_old = runs_fold / "train_plain" / "weights" / "best.pt"

    # B) plain layout under data_base
    w_plain_new = data_base / str(fold) / "train" / "weights" / "best.pt"

    # C) backup weights_base layout (recommended for your backup folder)
    w_plain_backup = None
    if weights_base is not None:
        w_plain_backup = weights_base / str(fold) / "train" / "weights" / "best.pt"

    def pick_by_mode():
        if mode == "hard":
            return (w_hard, "train_hard") if w_hard.exists() else (None, None)
        if mode == "normal":
            return (w_norm, "train1") if w_norm.exists() else (None, None)
        if mode == "plain":
            # prefer backup/plain-new/plain-old
            if w_plain_backup is not None and w_plain_backup.exists():
                return (w_plain_backup, "train_plain")
            if w_plain_new.exists():
                return (w_plain_new, "train_plain")
            if w_plain_old.exists():
                return (w_plain_old, "train_plain")
            return (None, None)
        # auto: hard > normal > plain
        if w_hard.exists():
            return (w_hard, "train_hard")
        if w_norm.exists():
            return (w_norm, "train1")
        if w_plain_backup is not None and w_plain_backup.exists():
            return (w_plain_backup, "train_plain")
        if w_plain_new.exists():
            return (w_plain_new, "train_plain")
        if w_plain_old.exists():
            return (w_plain_old, "train_plain")
        return (None, None)

    return pick_by_mode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_base", required=True, help="โฟลเดอร์ที่มี fold/test/images เช่น ...\\output_k_fold_boost_nopca หรือ ...\\backup\\dataset\\output_k_fold_boost_nopca")
    ap.add_argument("--weights_base", default=None, help="(optional) โฟลเดอร์ที่มี weights แบบ plain เช่น ...\\backup\\dataset\\plain_output_k_fold_boost_nopca")
    ap.add_argument("--folds", nargs="+", default=["0", "1", "2", "3", "4"])
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.01)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument(
        "--mode",
        choices=["hard", "normal", "plain", "auto"],
        default="auto",
        help="เลือก weights: hard, normal, plain หรือ auto (auto: hard > normal > plain)",
    )
    ap.add_argument(
        "--vote",
        choices=["top1", "weighted"],
        default="top1",
        help="วิธีเลือก class จาก detections: top1(conf สูงสุด) หรือ weighted(sum conf ต่อคลาส)",
    )
    ap.add_argument("--conf_vote_min", type=float, default=0.25, help="threshold สำหรับ weighted vote")
    args = ap.parse_args()

    data_base = Path(args.data_base).resolve()
    weights_base = Path(args.weights_base).resolve() if args.weights_base else None

    per_fold_summaries = []
    y_true_all, y_pred_all = [], []

    for fold in args.folds:
        fold_dir = data_base / str(fold)
        test_images_dir = fold_dir / "test" / "images"
        if not test_images_dir.exists():
            print(f"[fold {fold}] test/images not found: {test_images_dir}")
            continue

        # try read class ids from dataset.yaml (if exists)
        dataset_yaml = fold_dir / "dataset.yaml"
        nc, class_ids = read_classes_from_dataset_yaml(dataset_yaml)

        weights, model_name = find_weights_for_fold(
            fold=str(fold),
            mode=args.mode,
            data_base=data_base,
            weights_base=weights_base,
        )

        if weights is None:
            print(f"[fold {fold}] no weights found (mode={args.mode})")
            continue

        print(f"[fold {fold}] using weights ({model_name}): {weights}")
        imgs = collect_images(test_images_dir)
        print(f"[fold {fold}] #test images found: {len(imgs)}")

        model = YOLO(str(weights))

        rows = []
        infer_times = []
        total_times = []
        pre_ms_list = []
        post_ms_list = []

        for img in imgs:
            gt_path = img_to_gt_label_path(img)
            gt_cls = read_first_gt_class(gt_path)
            if gt_cls is None:
                continue

            t0 = time.perf_counter()

            res = model.predict(
                str(img),
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )

            t1 = time.perf_counter()

            r0 = res[0]
            if args.vote == "weighted":
                pred_cls = pick_image_label_weighted_vote(
                    r0,
                    conf_vote_min=args.conf_vote_min,
                    allowed_classes=set(class_ids) if class_ids is not None else None,
                )
            else:
                pred_cls = pick_image_label_from_detections(r0)

            speed = getattr(r0, "speed", {}) or {}
            pre_ms = speed.get("preprocess", None)
            inf_ms = speed.get("inference", None)
            post_ms = speed.get("postprocess", None)
            total_ms = (t1 - t0) * 1000.0

            infer_times.append(inf_ms)
            total_times.append(total_ms)
            pre_ms_list.append(pre_ms)
            post_ms_list.append(post_ms)

            rows.append(
                {
                    "fold": int(fold),
                    "image": str(img),
                    "y_true": int(gt_cls),
                    "y_pred": int(pred_cls),
                    "missed": int(pred_cls == -1),
                    "preprocess_ms": pre_ms,
                    "inference_ms": inf_ms,
                    "postprocess_ms": post_ms,
                    "total_ms": total_ms,
                }
            )

            y_true_all.append(int(gt_cls))
            y_pred_all.append(int(pred_cls))

        if not rows:
            print(f"[fold {fold}] no valid test samples (missing labels?)")
            continue

        df = pd.DataFrame(rows)

        # infer classes from dataset.yaml OR from y_true
        if class_ids is None:
            class_ids = sorted(df["y_true"].unique().tolist())

        # metrics (macro over true classes only; -1 counts as wrong and hurts recall)
        missed = int(df["missed"].sum())
        miss_rate = float(missed / len(df))

        y_true = df["y_true"].to_numpy()
        y_pred = df["y_pred"].to_numpy()

        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, labels=class_ids, average="macro", zero_division=0))
        rec = float(recall_score(y_true, y_pred, labels=class_ids, average="macro", zero_division=0))
        f1 = float(f1_score(y_true, y_pred, labels=class_ids, average="macro", zero_division=0))

        infer_np = np.array([t for t in infer_times if t is not None], dtype=float) if any(t is not None for t in infer_times) else None
        pre_np = np.array([t for t in pre_ms_list if t is not None], dtype=float) if any(t is not None for t in pre_ms_list) else None
        post_np = np.array([t for t in post_ms_list if t is not None], dtype=float) if any(t is not None for t in post_ms_list) else None
        total_np = np.array(total_times, dtype=float)

        def mean_std(a):
            if a is None or len(a) == 0:
                return (None, None)
            if len(a) == 1:
                return (float(np.mean(a)), 0.0)
            return (float(np.mean(a)), float(np.std(a, ddof=1)))

        pre_mean, pre_sd = mean_std(pre_np)
        inf_mean, inf_sd = mean_std(infer_np)
        post_mean, post_sd = mean_std(post_np)
        total_mean, total_sd = mean_std(total_np)

        print(
            f"[fold {fold}] n_images={len(df)}, missed={missed} ({miss_rate:.3f}), "
            f"acc={acc:.4f}, prec_macro={prec:.4f}, rec_macro={rec:.4f}, f1_macro={f1:.4f}, "
            f"infer_ms={inf_mean if inf_mean is not None else 'NA'}"
        )

        # output directory (keep compatible with both structures)
        # put eval next to weights folder if possible, else next to data_base
        out_dir = weights.parent.parent / "eval_test_official" if weights is not None else (data_base / "eval_test_official")
        out_dir.mkdir(parents=True, exist_ok=True)

        per_fold_csv = out_dir / f"per_image_{model_name}_fold{fold}.csv"
        df.to_csv(per_fold_csv, index=False, encoding="utf-8")
        print(f"[fold {fold}] per-image saved -> {per_fold_csv}")

        # confusion matrix (include -1 if present)
        unique_labels = sorted(set(df["y_true"].tolist()) | set(df["y_pred"].tolist()))
        cm = confusion_matrix(df["y_true"], df["y_pred"], labels=unique_labels)
        cm_df = pd.DataFrame(
            cm,
            index=[f"true_{c}" for c in unique_labels],
            columns=[f"pred_{c}" for c in unique_labels],
        )
        cm_csv = out_dir / f"confusion_matrix_{model_name}_fold{fold}.csv"
        cm_df.to_csv(cm_csv, encoding="utf-8")
        print(f"[fold {fold}] confusion matrix saved -> {cm_csv}")

        summary = {
            "fold": int(fold),
            "weights": str(weights),
            "model_name": model_name,
            "n_images": int(len(df)),
            "n_classes": int(len(class_ids)),
            "classes": class_ids,
            "missed(-1)": int(missed),
            "miss_rate": float(miss_rate),
            "accuracy": float(acc),
            "precision_macro": float(prec),
            "recall_macro": float(rec),
            "f1_macro": float(f1),
            "preprocess_ms_mean": pre_mean,
            "preprocess_ms_sd": pre_sd,
            "inference_ms_mean": inf_mean,
            "inference_ms_sd": inf_sd,
            "postprocess_ms_mean": post_mean,
            "postprocess_ms_sd": post_sd,
            "total_ms_mean": total_mean,
            "total_ms_sd": total_sd,
        }
        (out_dir / f"summary_{model_name}_fold{fold}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        per_fold_summaries.append(summary)

    # ---------- overall summary ----------
    if not per_fold_summaries:
        print("No folds evaluated.")
        return

    sum_df = pd.DataFrame(per_fold_summaries)

    # save overall summary next to data_base (stable)
    sum_csv = data_base / "eval_summary_test_official.csv"
    sum_df.to_csv(sum_csv, index=False, encoding="utf-8")
    print(f"\n==> summary of all folds saved -> {sum_csv}")

    # print mean ± sd over folds (unweighted)
    def mean_sd(vals):
        vals = np.array(vals, dtype=float)
        if len(vals) == 1:
            return float(np.mean(vals)), 0.0
        return float(np.mean(vals)), float(np.std(vals, ddof=1))

    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro",
               "inference_ms_mean", "total_ms_mean"]
    print("\n==== Mean ± SD over folds (unweighted) ====")
    for m in metrics:
        col = sum_df[m].dropna()
        if len(col) == 0:
            print(f"{m:22s}: NA")
            continue
        mu, sd = mean_sd(col.values)
        print(f"{m:22s}: {mu:.4f} ± {sd:.4f}")

    # weighted mean by n_images
    print("\n==== Weighted mean over folds (by n_images) ====")
    w = sum_df["n_images"].astype(float).to_numpy()
    for m in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]:
        vals = sum_df[m].astype(float).to_numpy()
        wmean = float(np.average(vals, weights=w))
        print(f"{m:22s}: {wmean:.4f}")


if __name__ == "__main__":
    print("EVAL SCRIPT:", os.path.abspath(__file__))
    main()

