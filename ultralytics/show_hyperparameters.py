# show_hyperparameters.py
# สคริปต์สำหรับดึงและแสดงค่า Hyperparameters หลักของแต่ละ Dataset ที่เทรนเสร็จแล้ว

import os
import yaml
import pandas as pd
from pathlib import Path

def main():
    # 1. กำหนดโฟลเดอร์หลักและรายชื่อ Dataset ที่ต้องการเช็ก
    root_dir = Path(r"C:\yolo8_strip\ultralytics")
    datasets = ["iris", "wine", "breast_cancer", "automobile", "hcc"] 
    
    records = []
    
    print("⏳ กำลังค้นหาไฟล์ Hyperparameters (args.yaml) ของแต่ละ Dataset...")
    
    for ds in datasets:
        ds_dir = root_dir / ds
        
        if not ds_dir.exists():
            records.append({
                "Dataset": ds, "Status": "❌ ไม่พบโฟลเดอร์", 
                "Epochs": "-", "Batch": "-", "Img_Size": "-", "LR(lr0)": "-", "Optimizer": "-"
            })
            continue

        # 2. ค้นหาไฟล์ args.yaml ในโฟลเดอร์ (ใช้ rglob ค้นหาทะลุโฟลเดอร์ย่อย)
        # โดยปกติ Hyperparameter ใน Dataset เดียวกันจะเหมือนกันทุก Fold เราจึงดึงแค่ไฟล์แรกที่เจอมาโชว์
        yaml_files = list(ds_dir.rglob("args.yaml"))
        
        if yaml_files:
            target_yaml = yaml_files[0] # หยิบไฟล์แรกที่เจอมาเป็นตัวแทน
            try:
                with open(target_yaml, 'r', encoding='utf-8') as f:
                    args = yaml.safe_load(f)
                    records.append({
                        "Dataset": ds,
                        "Status": "✅ พบข้อมูล",
                        "Epochs": args.get("epochs", "-"),
                        "Batch": args.get("batch", "-"),
                        "Img_Size": args.get("imgsz", "-"),
                        "LR(lr0)": args.get("lr0", "-"),
                        "Optimizer": args.get("optimizer", "-")
                    })
            except Exception as e:
                records.append({
                    "Dataset": ds, "Status": f"❌ อ่านไฟล์ไม่ได้", 
                    "Epochs": "-", "Batch": "-", "Img_Size": "-", "LR(lr0)": "-", "Optimizer": "-"
                })
        else:
            records.append({
                "Dataset": ds, "Status": "⚠️ ไม่พบ args.yaml (ยังไม่ได้เทรน)", 
                "Epochs": "-", "Batch": "-", "Img_Size": "-", "LR(lr0)": "-", "Optimizer": "-"
            })

    # 3. นำข้อมูลมาสร้างเป็นตารางด้วย Pandas
    df = pd.DataFrame(records)
    
    print("\n" + "="*80)
    print("📊 สรุปตาราง HYPERPARAMETERS ของแต่ละ DATASET")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)

if __name__ == "__main__":
    main()