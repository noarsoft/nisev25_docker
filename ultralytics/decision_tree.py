import os
import time
import warnings
import statistics

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import lib

import matplotlib.pyplot as plt
import seaborn as sns  # ตอนนี้ยังไม่ใช้ แต่เผื่ออยาก plot cm ทีหลัง

warnings.filterwarnings("ignore", message="X does not have valid feature names")

# ----------------- Config dataset -----------------
# ถ้าอยากเปลี่ยน dataset ให้เปลี่ยนฟังก์ชันด้านล่าง
_output_directory = os.path.dirname(os.path.abspath(__file__))

# result = [ [train_csv, val_csv, test_csv], ... ] แบบที่ print มาแล้ว
# label_name คือชื่อคอลัมน์ label (เช่น "income")
# เงื่อนไข: ก่อนรัน eval แนะนำให้รัน make_kfold_csv.py ให้เรียบร้อยแล้ว
#   python make_kfold_csv.py --dataset adult
# result, label_name = lib.load_k_float_encoder_adult(_output_directory)
# result, label_name = lib.load_k_float_encoder_iris(_output_directory)
# result, label_name = lib.load_k_float_encoder_wine(_output_directory)
# result, label_name = lib.load_k_float_encoder_automobile(_output_directory)
# result, label_name = lib.load_k_float_encoder_breast_cancer(_output_directory)
# result, label_name = lib.load_k_float_encoder_ionosphere(_output_directory)
result, label_name = lib.load_k_float_encoder_amphibians(_output_directory)

# print(result, label_name)

# exit(0)

# etc.

print("label column:", label_name)
print("k-fold csv list:")
for r in result:
    print("  train:", r[0])
    print("  val  :", r[1])
    print("  test :", r[2])
    print("-----")

# ----------------- Eval loop -----------------

accuracy_list = []
precision_list = []
recall_list = []
f1_list = []

per_fold_avg_times = []   # เก็บ average predict time ของแต่ละ fold (seconds per sample)

fold_idx = 0

for result_each in result:
    train_out, val_out, test_out = result_each

    train_data = pd.read_csv(train_out)
    val_data = pd.read_csv(val_out)
    test_data = pd.read_csv(test_out)

    # แยก X/y
    X_train = train_data.drop(columns=[label_name])
    y_train = train_data[label_name]

    X_val = val_data.drop(columns=[label_name])
    y_val = val_data[label_name]

    X_test = test_data.drop(columns=[label_name])
    y_test = test_data[label_name]

    # ----------------- จุดสำคัญ: รวม train + val -----------------
    X_trainval = pd.concat([X_train, X_val], ignore_index=True)
    y_trainval = pd.concat([y_train, y_val], ignore_index=True)

    # สามารถตั้ง hyperparameters เองได้
    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=None,      # ถ้าอยากจำกัดความลึก เช่น 10 ก็เปลี่ยนเป็น 10
        random_state=42
    )

    # ฝึกบน train+val
    t0 = time.time()
    model.fit(X_trainval, y_trainval)
    t1 = time.time()
    print(f"[fold {fold_idx}] fit time: {t1 - t0:.4f} sec, samples train+val = {len(X_trainval)}")

    # ----------------- วัดเวลา predict บน test -----------------
    t_pred_start = time.time()
    y_test_pred = model.predict(X_test)
    t_pred_end = time.time()

    avg_time_this_fold = (t_pred_end - t_pred_start) / len(X_test)
    per_fold_avg_times.append(avg_time_this_fold)
    print(f"[fold {fold_idx}] avg predict time: {avg_time_this_fold:.8f} sec/sample, test size = {len(X_test)}")

    # ----------------- คำนวณ metric -----------------
    # ใช้ evaluate_model ใน lib เพื่อให้รูปแบบเหมือนเดิม
    accuracy, precision, recall, f1, cm = lib.evaluate_model(y_test, y_test_pred)

    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)

    fold_idx += 1

# ----------------- สรุปผลข้ามทุก fold -----------------

# average predict time ทุก fold (เฉลี่ยของ avg per-fold)
overall_avg_time = sum(per_fold_avg_times) / len(per_fold_avg_times)
print("\n=== Prediction time over all folds ===")
print(f"Average prediction time per sample: {overall_avg_time:.8f} sec/sample")

# ใช้ calculate_std ที่มีอยู่แล้ว
mean_accuracy, std_accuracy = lib.calculate_std(accuracy_list)
mean_precision, std_precision = lib.calculate_std(precision_list)
mean_recall, std_recall = lib.calculate_std(recall_list)
mean_f1, std_f1 = lib.calculate_std(f1_list)

print("\n=== Metrics Over All Folds ===")
print(f"Mean Accuracy : {mean_accuracy:.4f} ± {std_accuracy:.4f}")
print(f"Mean Precision: {mean_precision:.4f} ± {std_precision:.4f}")
print(f"Mean Recall   : {mean_recall:.4f} ± {std_recall:.4f}")
print(f"Mean F1-Score : {mean_f1:.4f} ± {std_f1:.4f}")
