# eval_folds_test2.py
# 
# Example Usage:
# python eval_folds_test2.py --datasets iris --k_list 0,2,4,8,16 --device 0
# python eval_folds_test2.py --datasets wine --k_list 0,2,4,8,16 --device 0
# python eval_folds_test2.py --datasets breast_cancer --k_list 0,2,4,8,16 --device 0
# python eval_folds_test2.py --datasets automobile --k_list 0,2,4,8,16 --device 0
# python eval_folds_test2.py --datasets hcc --k_list 0,2,4,8,16 --device 0
# python eval_folds_test2.py --datasets ionosphere --k_list 0,2,4,8,16 --device 0

# python eval_folds_test2.py --datasets amphibians_brown_frogs --k_list 8 --device 0
# python eval_folds_test2.py --datasets amphibians_common_newt --k_list 8 --device 0
# python eval_folds_test2.py --datasets amphibians_common_toad --k_list 8 --device 0
# python eval_folds_test2.py --datasets amphibians_fire_bellied_toad --k_list 8 --device 0
# python eval_folds_test2.py --datasets amphibians_great_crested_newt --k_list 8 --device 0
# python eval_folds_test2.py --datasets amphibians_green_frogs --k_list 8 --device 0
# python eval_folds_test2.py --datasets amphibians_tree_frog --k_list 8 --device 0
import os
import time
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ultralytics import YOLO
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ชุดชื่อโฟลเดอร์ Algorithm (เอาไว้จับคู่หาไฟล์ Test และ Weight)
TARGET_FOLDERS = [
    "output_k_fold_boost_nopca",
    "output_k_fold_boost_nopca2",
    "output_k_fold_boost_nopca3",
    "output_k_fold_boost_pca",
    "output_k_fold_boost_pca2",
    "output_k_fold_boost_pca3",
]

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
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return -1
    confs = boxes.conf.cpu().tolist()
    clss = boxes.cls.cpu().tolist()
    k = int(max(range(len(confs)), key=lambda i: confs[i]))
    return int(clss[k])

def read_classes_from_dataset_yaml(dataset_yaml: Path):
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

def main():
    ap = argparse.ArgumentParser(description="Evaluate YOLO models across all K-lists and Algorithms")
    ap.add_argument("--dataset_root", default=r"C:\yolo8_strip\ultralytics", help="โฟลเดอร์หลักของโปรเจกต์")
    ap.add_argument("--datasets", required=True, help="ชื่อชุดข้อมูล เช่น iris, adult")
    ap.add_argument("--k_list", required=True, help="ค่า K ที่ต้องการประเมิน เช่น 0,2,4,8,16")
    ap.add_argument("--folds", nargs="+", default=["0", "1", "2", "3", "4"])
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.01)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    k_values = [x.strip() for x in args.k_list.split(",") if x.strip()]

    for ds in datasets:
        ds_dir = dataset_root / ds
        if not ds_dir.exists():
            print(f"❌ Dataset directory not found: {ds_dir}")
            continue

        for k in k_values:
            k_folder = ds_dir / f"repacking_k_{k}"
            if not k_folder.exists():
                print(f"\n[Skip] Repacking folder not found for K={k}: {k_folder}")
                continue

            print(f"\n" + "="*50)
            print(f"📊 EVALUATING DATASET: {ds} | REPACKING K={k}")
            print("="*50)

            for algo in TARGET_FOLDERS:
                data_base = k_folder / algo
                weights_base = k_folder / f"plain_{algo}"

                if not data_base.exists() or not weights_base.exists():
                    continue
                
                print(f"\n--- 🤖 Algorithm: {algo} ---")
                per_fold_summaries = []

                # ที่เก็บไฟล์สรุปผลของ Algorithm นี้
                eval_results_dir = weights_base / "eval_results_summary"
                eval_results_dir.mkdir(parents=True, exist_ok=True)

                for fold in args.folds:
                    fold_dir = data_base / str(fold)
                    test_images_dir = fold_dir / "test" / "images"
                    
                    if not test_images_dir.exists():
                        continue

                    dataset_yaml = fold_dir / "dataset.yaml"
                    nc, class_ids = read_classes_from_dataset_yaml(dataset_yaml)

                    # ตามหาไฟล์สมอง (best.pt) ในโฟลเดอร์ plain_...
                    weights = weights_base / str(fold) / "train" / "weights" / "best.pt"
                    
                    if not weights.exists():
                        print(f"  [Fold {fold}] ❌ Weights not found: {weights.parent.parent.name}")
                        continue

                    imgs = collect_images(test_images_dir)
                    if not imgs:
                        continue

                    print(f"  [Fold {fold}] ⏳ Evaluating {len(imgs)} test images...")

                    # โหลดโมเดล YOLO
                    model = YOLO(str(weights))

                    rows = []
                    infer_times = []
                    total_times = []

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

                        pred_cls = pick_image_label_from_detections(res[0])

                        # แกะเวลา Inference
                        speed = getattr(res[0], "speed", {}) or {}
                        inf_ms = speed.get("inference", None)
                        total_ms = (t1 - t0) * 1000.0

                        infer_times.append(inf_ms)
                        total_times.append(total_ms)

                        rows.append({
                            "fold": int(fold),
                            "image": img.name,
                            "y_true": int(gt_cls),
                            "y_pred": int(pred_cls),
                            "missed": int(pred_cls == -1),
                            "inference_ms": inf_ms,
                            "total_ms": total_ms,
                        })

                    if not rows:
                        continue

                    df = pd.DataFrame(rows)
                    if class_ids is None:
                        class_ids = sorted(df["y_true"].unique().tolist())

                    # คำนวณ Metrics (ถือว่าทายพลาด -1 คือความผิดพลาด)
                    missed = int(df["missed"].sum())
                    y_true = df["y_true"].to_numpy()
                    y_pred = df["y_pred"].to_numpy()

                    acc = float(accuracy_score(y_true, y_pred))
                    prec = float(precision_score(y_true, y_pred, labels=class_ids, average="macro", zero_division=0))
                    rec = float(recall_score(y_true, y_pred, labels=class_ids, average="macro", zero_division=0))
                    f1 = float(f1_score(y_true, y_pred, labels=class_ids, average="macro", zero_division=0))

                    infer_np = np.array([t for t in infer_times if t is not None], dtype=float) if any(t is not None for t in infer_times) else None
                    inf_mean = float(np.mean(infer_np)) if infer_np is not None else 0.0

                    print(f"      ✅ Acc={acc:.4f} | F1={f1:.4f} | Speed={inf_mean:.2f}ms")

                    # สร้างตาราง Confusion Matrix
                    unique_labels = sorted(set(df["y_true"].tolist()) | set(df["y_pred"].tolist()))
                    cm = confusion_matrix(df["y_true"], df["y_pred"], labels=unique_labels)
                    cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in unique_labels], columns=[f"pred_{c}" for c in unique_labels])
                    cm_df.to_csv(eval_results_dir / f"confusion_matrix_fold{fold}.csv", encoding="utf-8")
                    
                    df.to_csv(eval_results_dir / f"per_image_fold{fold}.csv", index=False, encoding="utf-8")

                    per_fold_summaries.append({
                        "fold": int(fold),
                        "algorithm": algo,
                        "n_images": int(len(df)),
                        "missed_detections": missed,
                        "accuracy": acc,
                        "precision_macro": prec,
                        "recall_macro": rec,
                        "f1_macro": f1,
                        "inference_ms_mean": inf_mean,
                    })

                # สรุปผลรวม 5 Fold ของ Algorithm นี้
                if per_fold_summaries:
                    sum_df = pd.DataFrame(per_fold_summaries)
                    sum_csv_path = eval_results_dir / "00_SUMMARY_ALL_FOLDS.csv"
                    sum_df.to_csv(sum_csv_path, index=False, encoding="utf-8")
                    
                    # คำนวณค่าเฉลี่ย
                    acc_mean = sum_df["accuracy"].mean()
                    f1_mean = sum_df["f1_macro"].mean()
                    print(f"   🎯 [Summary {algo}] Avg Acc: {acc_mean:.4f} | Avg F1: {f1_mean:.4f}")
                    print(f"   💾 Saved to: {eval_results_dir.relative_to(dataset_root)}")

if __name__ == "__main__":
    main()