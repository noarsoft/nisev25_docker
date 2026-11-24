import os
import argparse
import pandas as pd
from pathlib import Path

# python compare_multilabel.py --r_list 8

TARGET_FOLDERS = [
    "output_k_fold_boost_nopca",
    "output_k_fold_boost_pca",
]

AMPHIBIANS_LABELS = [
    "amphibians_green_frogs",
    "amphibians_brown_frogs",
    "amphibians_common_toad",
    "amphibians_fire_bellied_toad",
    "amphibians_tree_frog",
    "amphibians_common_newt",
    "amphibians_great_crested_newt"
]

BASELINE_F1_CUTOFF = 0.5000

def main():
    ap = argparse.ArgumentParser(description="Detailed Multi-label Evaluation for Amphibians (7 Models + Macro Average)")
    ap.add_argument("--dataset_root", default=r"C:\yolo8_strip\ultralytics", help="Root directory")
    ap.add_argument("--r_list", required=True, help="Comma-separated r values e.g. 4")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    r_values = [x.strip() for x in args.r_list.split(",") if x.strip()]

    detailed_records = []
    macro_records = []

    print(f"\n🔍 กำลังตรวจสอบผลการประเมินของทั้ง 7 โมเดลย่อยสำหรับชุดข้อมูล Amphibians ...\n")

    for r in r_values:
        for algo in TARGET_FOLDERS:
            label_metrics = []
            
            print(f"--- Algorithm: {algo} (r={r}) ---")
            for label_ds in AMPHIBIANS_LABELS:
                k_folder = dataset_root / label_ds / f"repacking_k_{r}"
                summary_csv = k_folder / f"plain_{algo}" / "eval_results_summary" / "00_SUMMARY_ALL_FOLDS.csv"
                
                if not summary_csv.exists():
                    print(f"  ❌ [{label_ds}] ไม่พบไฟล์สรุปผล")
                    continue
                
                df = pd.read_csv(summary_csv)
                if not df.empty:
                    acc = df["accuracy"].mean()
                    prec = df["precision_macro"].mean()
                    rec = df["recall_macro"].mean()
                    f1 = df["f1_macro"].mean()
                    
                    # 💡 Balanced Accuracy = (Sensitivity + Specificity) / 2
                    # ซึ่งใน Binary Classification ค่า Macro Recall จะมีค่าเท่ากับสูตรนี้พอดี
                    bal_acc = rec 
                    
                    print(f"  ✅ [{label_ds}] Acc: {acc:.4f} | Bal_Acc: {bal_acc:.4f} | F1: {f1:.4f}")
                    
                    # เก็บข้อมูลแยกตามโมเดล (Individual Model Record)
                    detailed_records.append({
                        "Model_Class": label_ds,
                        "Algorithm": algo,
                        "r_Value": int(r),
                        "Accuracy": acc,
                        "Balanced_Accuracy": bal_acc,  # เพิ่มคอลัมน์ Balanced Accuracy
                        "Precision": prec,
                        "Recall": rec,
                        "F1_Score": f1
                    })
                    
                    label_metrics.append({
                        "accuracy": acc,
                        "balanced_accuracy": bal_acc,
                        "precision": prec,
                        "recall": rec,
                        "f1": f1
                    })
            
            # ถ้ามีข้อมูลครบหรือบางส่วน นำมาคำนวณ Macro-Average
            if len(label_metrics) > 0:
                agg_df = pd.DataFrame(label_metrics)
                macro_records.append({
                    "Algorithm": algo,
                    "r_Value": int(r),
                    "Models_Count": len(label_metrics),
                    "Accuracy_Mean": agg_df["accuracy"].mean(),
                    "Balanced_Acc_Mean": agg_df["balanced_accuracy"].mean(), # เพิ่มค่าเฉลี่ย Balanced Accuracy
                    "Precision_Mean": agg_df["precision"].mean(),
                    "Recall_Mean": agg_df["recall"].mean(),
                    "F1_Score_Mean": agg_df["f1"].mean()
                })
            print("-" * 50)

    if not detailed_records:
        print("\n❌ ไม่พบข้อมูลผลการประเมินใดๆ กรุณารัน eval_folds_test2.py ให้ครบทุกคลาสก่อนครับ")
        return

    # บันทึกและแสดงผลลัพธ์แบบรายโมเดล (Individual Models)
    detailed_df = pd.DataFrame(detailed_records)
    detailed_csv = dataset_root / "DETAILED_7MODELS_amphibians.csv"
    detailed_df.to_csv(detailed_csv, index=False, encoding="utf-8-sig")

    # บันทึกและแสดงผลลัพธ์ภาพรวม (Macro-Average Summary)
    macro_df = pd.DataFrame(macro_records)
    macro_csv = dataset_root / "MACRO_SUMMARY_amphibians_multilabel.csv"
    macro_df.to_csv(macro_csv, index=False, encoding="utf-8-sig")

    print("\n" + "="*90)
    print("📋 1. ผลลัพธ์แยกตามแต่ละโมเดลย่อย (Individual Models Performance)")
    print("="*90)
    print(detailed_df.to_string(index=False))
    print(f"\n💾 บันทึกไฟล์รายละเอียดรายโมเดลที่: {detailed_csv}")

    print("\n" + "="*90)
    print("📊 2. ค่าเฉลี่ยรวม (Macro-Average Summary ทั้งหมด)")
    print("="*90)
    print(macro_df.to_string(index=False))
    print(f"\n💾 บันทึกไฟล์สรุปภาพรวมที่: {macro_csv}")

    if not macro_df.empty:
        best_row = macro_df.loc[macro_df['F1_Score_Mean'].idxmax()]
        print("\n" + "-"*35)
        print(" ประกาศผล: Algorithm ภาพรวมที่ดีที่สุด (Macro-Average)")
        print(f"   🔹 Algorithm : {best_row['Algorithm']}")
        print(f"   🔹 r_Value   : {int(best_row['r_Value'])}")
        print(f"   🔹 Models    : {int(best_row['Models_Count'])} / 7 models")
        print(f"   🔹 Accuracy  : {best_row['Accuracy_Mean']:.4f}")
        print(f"   🔹 Bal_Acc   : {best_row['Balanced_Acc_Mean']:.4f}")
        print(f"   🔹 F1-Score  : {best_row['F1_Score_Mean']:.4f}")
        print("-"*35)
        
        print(f"\n[Baseline Evaluation - Cut-off at {BASELINE_F1_CUTOFF * 100:.0f}%]")
        if best_row['F1_Score_Mean'] > BASELINE_F1_CUTOFF:
            print("   ✅ SUCCESS: โมเดลทดลองทำคะแนนเฉลี่ยสูงกว่า Baseline (50%)!")
        else:
            print("   ❌ FAILED: โมเดลทดลองยังไม่สามารถผ่านเกณฑ์ Baseline (50%) ได้")

if __name__ == "__main__":
    main()