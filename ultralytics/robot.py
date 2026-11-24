# robot.py
import os
import time
import json
import psycopg2
from dotenv import load_dotenv
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

# 1. โหลดค่าคอนฟิก
load_dotenv()
WORKSPACE_DIR = Path("bot_workspace")
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

# 2. ตั้งค่า YOLO
DATASET_ROOT = Path("C:/yolo8_strip/ultralytics")
DATASET_NAME = "iris"                          
K_VALUE = 4                                    
ALGO = "plain_output_k_fold_boost_nopca"       
FOLD = 0                                       
MODEL_PATH = DATASET_ROOT / DATASET_NAME / f"repacking_k_{K_VALUE}" / ALGO / str(FOLD) / "train" / "weights" / "best.pt"

print("========================================")
print("🤖 กำลังเปิดระบบบอท (Robot Worker)...")
if MODEL_PATH.exists():
    model = YOLO(str(MODEL_PATH))
    print("✅ โหลดโมเดล YOLO สำเร็จ")
else:
    print(f"❌ ไม่พบโมเดลที่: {MODEL_PATH}")
    model = None

# โหลด Min-Max Data
CSV_SOURCE_PATH = DATASET_ROOT / DATASET_NAME / f"{DATASET_NAME}.csv"
features_min_max = {}
csv_feature_order = []
if CSV_SOURCE_PATH.exists():
    df_src = pd.read_csv(CSV_SOURCE_PATH)
    df_src.columns = [str(c).replace('_', '.') for c in df_src.columns]
    for col in df_src.columns:
        if pd.api.types.is_numeric_dtype(df_src[col]):
            features_min_max[col] = {"min": float(df_src[col].min()), "max": float(df_src[col].max())}
            csv_feature_order.append(col)

# 3. ฟังก์ชันสร้างรูป
def generate_strip_image(features_dict, save_path):
    normalized_features = {k.replace('_', '.'): v for k, v in features_dict.items()}
    feature_order = [col for col in csv_feature_order if col in normalized_features]
    if not feature_order: feature_order = list(normalized_features.keys())

    pixel_values = []
    for col in feature_order:
        val = normalized_features.get(col, 0.0)
        limits = features_min_max.get(col, {"min": 0.0, "max": 10.0})
        if limits["max"] == limits["min"]: scaled_val = 0
        else:
            scaled_val = int(((val - limits["min"]) / (limits["max"] - limits["min"])) * 255)
            scaled_val = max(0, min(255, scaled_val))
        pixel_values.append(scaled_val)
        
    img_arr = np.array([pixel_values], dtype=np.uint8)
    img = Image.fromarray(img_arr, mode='L').convert('RGB')
    canvas = Image.new("RGB", (640, 640), (255, 255, 255))
    cell_size = 40
    tile = img.resize((cell_size, cell_size), Image.Resampling.NEAREST)
    canvas.paste(tile, ((640 - cell_size) // 2, (640 - cell_size) // 2))
    canvas.save(save_path, "JPEG")

# 4. ฟังก์ชันเชื่อม DB
def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT"),
        database=os.getenv("DB_NAME"), user=os.getenv("DB_USER"), password=os.getenv("DB_PASSWORD")
    )

# 5. ลูปการทำงานหลักของ Bot
def start_bot():
    print("⏳ บอทพร้อมทำงาน กำลังรอคิวจาก Database...")
    while True:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # ค้นหา 1 คิวที่ pending 
            cursor.execute("""
                UPDATE queue 
                SET queue_status = 'processing', updated_at = CURRENT_TIMESTAMP
                WHERE queue_id = (
                    SELECT queue_id FROM queue 
                    WHERE queue_status = 'pending' 
                    ORDER BY queue_id ASC 
                    LIMIT 1
                    FOR UPDATE SKIP LOCKED
                )
                RETURNING queue_id, payload;
            """)
            
            task = cursor.fetchone()
            
            if task:
                queue_id = task[0]
                payload = task[1] 
                print(f"🔄 พบงานใหม่! Queue ID: {queue_id}")
                
                # --- เริ่มกระบวนการ YOLO ---
                image_path = WORKSPACE_DIR / f"temp_{queue_id}.jpg"
                generate_strip_image(payload["features"], image_path)
                
                if model:
                    # ปรับ conf=0.001 เพื่อให้ดึงคลาสอันดับรองๆ ที่อาจจะมีความมั่นใจน้อยติดมาด้วย
                    results = model.predict(source=str(image_path), conf=0.001, imgsz=640, verbose=False)
                    boxes = results[0].boxes
                    
                    if len(boxes) > 0:
                        confs = boxes.conf.cpu().tolist()
                        clss = boxes.cls.cpu().tolist()
                        
                        # รวบรวมคลาสและค่าความมั่นใจ (กรองเอาค่า max กรณีที่โมเดลหากล่องคลาสเดียวกันเจอหลายกล่อง)
                        class_scores = {}
                        for c, conf in zip(clss, confs):
                            c_int = int(c)
                            if c_int not in class_scores or conf > class_scores[c_int]:
                                class_scores[c_int] = float(conf)
                        
                        # สร้างรายการ Ranking และคำนวณเปอร์เซ็นต์
                        rankings = [
                            {
                                "class": k, 
                                "confidence": v, 
                                "percentage": round(v * 100, 2)
                            }
                            for k, v in class_scores.items()
                        ]
                        
                        # เรียงลำดับจากความมั่นใจมากไปน้อย
                        rankings = sorted(rankings, key=lambda x: x["confidence"], reverse=True)
                        
                        # ดึงตัวที่ชนะเลิศมาเป็นค่าหลัก
                        best_prediction = rankings[0]
                        predicted_class = best_prediction["class"]
                        confidence = best_prediction["confidence"]
                    else:
                        predicted_class = -1
                        confidence = 0.0
                        rankings = []
                else:
                    predicted_class = -1
                    confidence = 0.0
                    rankings = []
                
                # สร้างผลลัพธ์
                yolo_result = {
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "percentage": round(confidence * 100, 2) if confidence > 0 else 0.0,
                    "rankings": rankings,
                    "model_dataset": DATASET_NAME
                }
                
                # อัปเดตผลลัพธ์กลับลง DB
                cursor.execute("""
                    UPDATE queue 
                    SET queue_status = 'done', yolo_result = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE queue_id = %s;
                """, (json.dumps(yolo_result), queue_id))
                
                conn.commit()
                print(f"✅ ทำงานเสร็จสิ้น Queue ID: {queue_id} (Class: {predicted_class} | {yolo_result['percentage']}%)")
                
                # ลบรูปชั่วคราวทิ้ง (เพื่อประหยัดพื้นที่)
                # if image_path.exists():
                #     os.remove(image_path)
                    
            else:
                conn.rollback()
                time.sleep(1)
                
        except Exception as e:
            conn.rollback()
            print(f"❌ เกิดข้อผิดพลาด: {e}")
            time.sleep(5)
        finally:
            cursor.close()
            conn.close()

if __name__ == "__main__":
    start_bot()