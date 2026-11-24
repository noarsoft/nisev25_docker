# compare_k_results.py
# 
# Example Usage:
# python compare_k_results.py --datasets iris --k_list 0,2,4,8,16
# python compare_k_results.py --datasets wine --k_list 0,2,4,8,16
# python compare_k_results.py --datasets breast_cancer --k_list 0,2,4,8,16
# python compare_k_results.py --datasets automobile --k_list 0,2,4,8,16       
# python compare_k_results.py --datasets hcc --k_list 0,2,4,8,16
# python compare_k_results.py --datasets ionosphere --k_list 0,2,4,8,16
import os
import argparse
import pandas as pd
from pathlib import Path

# ชุดชื่อโฟลเดอร์ Algorithm
TARGET_FOLDERS = [
    "output_k_fold_boost_nopca",
    "output_k_fold_boost_nopca2",
    "output_k_fold_boost_nopca3",
    "output_k_fold_boost_pca",
    "output_k_fold_boost_pca2",
    "output_k_fold_boost_pca3",
]

# กำหนดเกณฑ์ Baseline Cut-off ที่ 70%
BASELINE_F1_CUTOFF = 0.50

def main():
    ap = argparse.ArgumentParser(description="สร้างตารางเปรียบเทียบประสิทธิภาพระหว่างค่า K ต่างๆ")
    ap.add_argument("--dataset_root", default=r"C:\yolo8_strip\ultralytics", help="โฟลเดอร์หลัก")
    ap.add_argument("--datasets", required=True, help="ชื่อชุดข้อมูล เช่น iris, adult")
    ap.add_argument("--k_list", required=True, help="ค่า K ที่ต้องการเปรียบเทียบ เช่น 0,2,4,8,16")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    k_values = [x.strip() for x in args.k_list.split(",") if x.strip()]

    for ds in datasets:
        ds_dir = dataset_root / ds
        if not ds_dir.exists():
            print(f" ไม่พบโฟลเดอร์ Dataset: {ds_dir}")
            continue

        master_records = []

        print(f"\n กำลังรวบรวมผลลัพธ์ของ Dataset: {ds} ...")

        for k in k_values:
            k_folder = ds_dir / f"repacking_k_{k}"
            if not k_folder.exists():
                continue

            for algo in TARGET_FOLDERS:
                summary_csv = k_folder / f"plain_{algo}" / "eval_results_summary" / "00_SUMMARY_ALL_FOLDS.csv"
                
                if not summary_csv.exists():
                    continue
                
                df = pd.read_csv(summary_csv)
                if df.empty:
                    continue

                record = {
                    "Dataset": ds,
                    "Algorithm": algo,
                    "K_Value": int(k),
                    "Accuracy_Mean": df["accuracy"].mean(),
                    "Accuracy_Std": df["accuracy"].std(),
                    "Precision_Mean": df["precision_macro"].mean(),
                    "Recall_Mean": df["recall_macro"].mean(),
                    "F1_Score_Mean": df["f1_macro"].mean(),
                    "F1_Score_Std": df["f1_macro"].std(),
                    "Inference_Speed_ms": df["inference_ms_mean"].mean()
                }
                master_records.append(record)

        if not master_records:
            print(f" ไม่พบข้อมูลสรุปผลสำหรับ Dataset: {ds} กรุณารัน eval_folds_test2.py ก่อน")
            continue

        master_df = pd.DataFrame(master_records)

        # เรียงลำดับโชว์ในตาราง
        master_df = master_df.sort_values(by=["Algorithm", "K_Value"])

        out_csv = ds_dir / f"MASTER_COMPARISON_{ds}.csv"
        master_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

        print("="*70)
        print(f" สร้างตารางเปรียบเทียบสำเร็จ! บันทึกไว้ที่:")
        print(f" {out_csv}")
        print("="*70)
        
        # ปริ้นท์ตารางทั้งหมด (เพิ่ม Precision และ Recall)
        print("\n" + master_df[["Algorithm", "K_Value", "Accuracy_Mean", "Precision_Mean", "Recall_Mean", "F1_Score_Mean", "Inference_Speed_ms"]].to_string(index=False))
        
        # ---------------------------------------------------------
        # ค้นหาผู้ชนะ (Best Model) อิงจาก F1-Score สูงสุด
        # ---------------------------------------------------------
        best_row = master_df.loc[master_df['F1_Score_Mean'].idxmax()]
        

        print("\n ประกาศผล: โมเดลที่ดีที่สุด (อิงจาก F1-Score) ")
        print(f"   🔹 Algorithm : {best_row['Algorithm']}")
        print(f"   🔹 K_Value   : {int(best_row['K_Value'])}")
        print(f"   🔹 Accuracy  : {best_row['Accuracy_Mean']:.4f} ({(best_row['Accuracy_Mean']*100):.2f}%)")
        print(f"   🔹 Precision : {best_row['Precision_Mean']:.4f} ({(best_row['Precision_Mean']*100):.2f}%)")
        print(f"   🔹 Recall    : {best_row['Recall_Mean']:.4f} ({(best_row['Recall_Mean']*100):.2f}%)")
        print(f"   🔹 F1-Score  : {best_row['F1_Score_Mean']:.4f} ({(best_row['F1_Score_Mean']*100):.2f}%)")
        print(f"   🔹 Speed     : {best_row['Inference_Speed_ms']:.2f} ms")

        # ---------------------------------------------------------
        # ตรวจสอบกับ Baseline Cut-off
        # ---------------------------------------------------------
        print(f"\n[Baseline Evaluation - Cut-off at {BASELINE_F1_CUTOFF * 100:.0f}%]")
        if best_row['F1_Score_Mean'] > BASELINE_F1_CUTOFF:
            print("   ✅ SUCCESS: โมเดลทดลองทำคะแนนได้สูงกว่า Baseline (50%)!")
        else:
            print("   ❌ FAILED: โมเดลทดลองยังไม่สามารถผ่านเกณฑ์ Baseline (50%) ได้")

if __name__ == "__main__":
    main()