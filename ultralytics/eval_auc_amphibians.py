import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from sklearn.metrics import roc_auc_score
# python eval_auc_amphibians.py --r_list 8

# กรองมาเฉพาะ Algorithm 1 และ 2 ตามที่คุณต้องการ
TARGET_FOLDERS = [
    "output_k_fold_boost_nopca",
    "output_k_fold_boost_pca",
]

# 7 คลาสของ Amphibians
AMPHIBIANS_LABELS = [
    "amphibians_green_frogs",
    "amphibians_brown_frogs",
    "amphibians_common_toad",
    "amphibians_fire_bellied_toad",
    "amphibians_tree_frog",
    "amphibians_common_newt",
    "amphibians_great_crested_newt"
]

def collect_images(root: Path):
    if not root.exists():
        return []
    return [p for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.is_file()]

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

def get_prob_class_1(result):
    """ ดึงค่าความน่าจะเป็น (Probability) ว่ารูปนี้คือคลาส 1 """
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return 0.0 # หาไม่เจอ ถือว่าความมั่นใจเป็น 0
    
    confs = boxes.conf.cpu().tolist()
    clss = boxes.cls.cpu().tolist()
    
    # ดึงกล่องที่โมเดลมั่นใจที่สุด
    k = int(max(range(len(confs)), key=lambda i: confs[i]))
    pred_cls = int(clss[k])
    conf = confs[k]
    
    # ถ้าทายเป็น 1 ก็ใช้ค่า conf ได้เลย ถ้าทายเป็น 0 ความน่าจะเป็นของ 1 คือ (1 - conf)
    if pred_cls == 1:
        return conf
    else:
        return 1.0 - conf

def main():
    ap = argparse.ArgumentParser(description="Calculate AUC for Amphibians Multi-label")
    ap.add_argument("--dataset_root", default=r"C:\yolo8_strip\ultralytics", help="Root directory")
    ap.add_argument("--r_list", required=True, help="Comma-separated r values e.g. 4")
    ap.add_argument("--device", default="0", help="CUDA Device")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    r_values = [x.strip() for x in args.r_list.split(",") if x.strip()]

    macro_records = []
    
    print("\n" + "="*60)
    print("📈 เริ่มกระบวนการคำนวณ AUC สำหรับชุดข้อมูล Amphibians (7 โมเดล)")
    print("="*60)

    for r in r_values:
        for algo in TARGET_FOLDERS:
            label_metrics = []
            print(f"\n🚀 Algorithm: {algo} | r={r}")
            
            for label_ds in AMPHIBIANS_LABELS:
                all_y_true = []
                all_y_prob = []
                
                k_folder = dataset_root / label_ds / f"repacking_k_{r}"
                data_base = k_folder / algo
                weights_base = k_folder / f"plain_{algo}"

                if not data_base.exists() or not weights_base.exists():
                    continue

                # รวบรวมข้อมูลจากทั้ง 5 Folds เพื่อคำนวณ AUC รวดเดียว
                for fold in range(5):
                    fold_dir = data_base / str(fold)
                    test_images_dir = fold_dir / "test" / "images"
                    weights = weights_base / str(fold) / "train" / "weights" / "best.pt"

                    if not test_images_dir.exists() or not weights.exists():
                        continue

                    imgs = collect_images(test_images_dir)
                    if not imgs:
                        continue

                    model = YOLO(str(weights))
                    
                    for img in imgs:
                        gt_path = img_to_gt_label_path(img)
                        gt_cls = read_first_gt_class(gt_path)
                        if gt_cls is None:
                            continue

                        res = model.predict(str(img), device=args.device, verbose=False)
                        prob_1 = get_prob_class_1(res[0])
                        
                        all_y_true.append(int(gt_cls))
                        all_y_prob.append(prob_1)

                # คำนวณ AUC สำหรับโมเดลนี้
                if len(set(all_y_true)) > 1: # ต้องมีทั้งคลาส 0 และ 1 ถึงจะคำนวณ AUC ได้
                    auc_score = roc_auc_score(all_y_true, all_y_prob)
                    label_metrics.append({"label": label_ds, "auc": auc_score})
                    print(f"  ✅ [{label_ds}] AUC: {auc_score:.4f}")
                else:
                    print(f"  ⚠️ [{label_ds}] ไม่สามารถคำนวณ AUC ได้ (มีเพียงคลาสเดียวใน Test Set)")

            # คำนวณ Macro-AUC (เฉลี่ยจาก 7 โมเดล)
            if len(label_metrics) > 0:
                macro_auc = np.mean([item["auc"] for item in label_metrics])
                macro_records.append({
                    "Algorithm": algo,
                    "r_Value": int(r),
                    "Evaluated_Models": len(label_metrics),
                    "Macro_AUC": macro_auc
                })
                print(f"  🎯 Macro-AUC ({algo}): {macro_auc:.4f}")

    if not macro_records:
        print("\n❌ ไม่สามารถคำนวณ AUC ได้ โปรดตรวจสอบว่าโมเดลได้ถูกเทรนเสร็จสมบูรณ์แล้ว")
        return

    # บันทึกไฟล์ผลลัพธ์
    df_auc = pd.DataFrame(macro_records)
    out_csv = dataset_root / "MACRO_AUC_amphibians.csv"
    df_auc.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\n" + "="*60)
    print(f"🎉 สรุปผลลัพธ์การคำนวณค่า AUC (Area Under the Curve)")
    print("="*60)
    print(df_auc.to_string(index=False))
    print(f"\n💾 บันทึกตารางสรุปผลไว้ที่: {out_csv}")

if __name__ == "__main__":
    main()