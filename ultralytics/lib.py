#lib.py
# python C:\yolov5_strip\train.py --data C:\yolov5_strip\output_k_fold3\3\dataset.yaml --weights C:\yolov5_strip\yolov5s.pt --img 640 --batch-size 2 --epochs 1000


import math
import statistics
import cv2  # pip install opencv-python
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import pandas as pd
import shutil
import os
from pathlib import Path
import glob
from collections import Counter
import json # <-- เพิ่ม json ไว้ตรงนี้ด้วยครับ

# ======== ครอบ try-except ป้องกัน DLL Block ========
try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
except ImportError as e:
    print(f"⚠️ ข้ามการโหลด sklearn เนื่องจากติด Policy ของเครื่อง (ระบบ Inference ยังใช้งานได้ปกติ)")
# ===================================================


_insert_text_to_img_mode = 0

def data_scatterplot2column(csv_file,col1,col2,outpath_jpg):

    data = pd.read_csv(csv_file)
    x_column = col1
    y_column = col2
    category_column = data.columns[-1]

    unique_categories = data[category_column].unique()
    color_map = {category: color for category, color in zip(unique_categories, plt.cm.tab10.colors)}

    plt.figure(figsize=(10, 8))

    for category, color in color_map.items():
        subset = data[data[category_column] == category]
        plt.scatter(subset[x_column], subset[y_column], label=category, color=color, alpha=0.7)

    plt.title(f"Scatterplot of {x_column} and {y_column}, label {category_column}")
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.legend(title=category_column)
    plt.grid(True)

    plt.savefig(outpath_jpg, format="jpg", dpi=300)  # Use dpi=300 for high resolution
    print(f"Scatterplot saved as {outpath_jpg}")

    plt.show()



def evaluate_model(y_true, y_pred):
    #accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    #precision
    precision = precision_score(y_true, y_pred, average='weighted')
    print(f"Precision: {precision:.2f}")

    #recall
    recall = recall_score(y_true, y_pred, average='weighted')
    print(f"Recall: {recall:.2f}")

    #f-score
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"F1-Score: {f1:.2f}")

    # print(y_true)

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=set(y_true), yticklabels=set(y_true))
    # plt.title('Confusion Matrix')
    # plt.xlabel('Predicted Labels')
    # plt.ylabel('True Labels')
    # plt.show()
    #

    # print("\nClassification Report:")
    # print(classification_report(y_true, y_pred, zero_division=0))


    return accuracy,precision,recall,f1,cm


def calculate_std(metrics_list):
    mean = sum(metrics_list) / len(metrics_list)
    std = statistics.stdev(metrics_list)
    return mean, std

def remove_and_create_folder(list_folder):
    for folder1 in list_folder:
        if os.path.exists(folder1):
            shutil.rmtree(folder1)
        os.makedirs(folder1)

def csv_to_pca_csv(project_directory, csv, label_column):
    # Load the CSV file
    df = pd.read_csv(csv)
    # Separating out the features and the label
    X = df.drop(columns=[label_column])
    y = df[label_column]

    X = StandardScaler().fit_transform(X)

    # PCA
    pca = PCA(n_components=2)  # Adjust n_components as needed
    principal_components = pca.fit_transform(X)

    # Convert the principal components to a DataFrame
    principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

    final_df = pd.concat([principal_df, y], axis=1)
    final_df.to_csv(os.path.join(project_directory, 'pca_result.csv'), index=False)
    return os.path.join(project_directory, 'pca_result.csv')

def df_to_pca_csv(project_directory, df, label_column):
    #Load the CSV file
    #Separating out the features and the label
    X1 = df.drop(columns=[label_column])
    X = df.drop(columns=[label_column])
    y = df[label_column]

    # nan_rows = df[df.isna().any(axis=1)]
    # print("Rows with NaN values:")
    # print(nan_rows)

    X = StandardScaler().fit_transform(X)

    #PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X)

    #Convert the principal components to a DataFrame
    principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

    final_df = pd.concat([principal_df, y], axis=1)
    final_df = pd.concat([principal_df, X1], axis=1)

    nan_rows = final_df[final_df.isna().any(axis=1)]
    print("Rows with NaN values:")
    print(nan_rows)


    final_df.to_csv(os.path.join(project_directory, 'pca_result.csv'), index=False)
    return os.path.join(project_directory, 'pca_result.csv')


def load_k_float_encoder_wine(project_directory):
    path_csv = os.path.join(project_directory, "wine.csv")
    label_name = "Wine"
    result = load_k_float_encoder_to_csv(project_directory, path_csv, dataset_name="wine")
    return result, label_name


def load_k_float_encoder_iris(project_directory):
    path_csv = os.path.join(project_directory, "iris.csv")
    label_name = "variety"
    result = load_k_float_encoder_to_csv(project_directory, path_csv, dataset_name="iris")
    return result, label_name


def load_k_float_encoder_yeast(project_directory):
    path_csv = os.path.join(project_directory, "yeast.csv")
    label_name = "name"
    result = load_k_float_encoder_to_csv(project_directory, path_csv)
    return result, label_name

def load_k_float_encoder_nutrition(project_directory):
    path_csv = os.path.join(project_directory, "nutrition.csv")
    label_name = "label"
    result = load_k_float_encoder_to_csv(project_directory, path_csv)
    return result, label_name


def load_k_float_encoder_adult(project_directory):
    path_csv = os.path.join(project_directory, "_adult.csv") # modify string and missing
    label_name = "income"
    result = load_k_float_encoder_to_csv(project_directory, path_csv, dataset_name="adult")
    return result, label_name


def load_k_float_encoder_electric(project_directory):
    path_csv = os.path.join(project_directory, "_electricity.csv") # modify string and missing
    label_name = "class"
    result = load_k_float_encoder_to_csv(project_directory, path_csv)
    return result, label_name


def load_k_float_encoder_student_performance(project_directory):
    path_csv = os.path.join(project_directory, "Student_performance_data.csv")
    label_name = "GradeClass"
    result = load_k_float_encoder_to_csv(project_directory, path_csv)
    return result, label_name

def load_k_float_encoder_automobile(project_directory):
    clean_csv = os.path.join(project_directory, "_automobile.csv")
    raw_csv = os.path.join(project_directory, "automobile.csv")
    
    # ถ้ายังไม่มีไฟล์คลีนที่หน้า Root ให้สร้างใหม่เลย
    if not os.path.exists(clean_csv):
        df = pd.read_csv(raw_csv)
        df.replace('?', np.nan, inplace=True)
        df.dropna(inplace=True)
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            
        df.to_csv(clean_csv, index=False)

    label_name = "symboling"
    result = load_k_float_encoder_to_csv(project_directory, clean_csv,dataset_name="automobile")
    return result, label_name


def load_k_float_encoder_hcc(project_directory):
    clean_csv = os.path.join(project_directory, "_hcc.csv")
    raw_csv = os.path.join(project_directory, "hcc.csv")
    
    if not os.path.exists(clean_csv):
        df = pd.read_csv(raw_csv, header=None)
        df.replace('?', np.nan, inplace=True)
        df.dropna(inplace=True)
        
        num_features = len(df.columns) - 1
        feature_cols = [f"feature_{i}" for i in range(1, num_features + 1)]
        df.columns = feature_cols + ["class_label"]
        
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)
        
        df.to_csv(clean_csv, index=False)

    label_name = "class_label"
    result = load_k_float_encoder_to_csv(project_directory, clean_csv,dataset_name="hcc")
    return result, label_name


def load_header_ionosphere(project_directory):
    import numpy as np
    import pandas as pd
    import os
    
    path_csv = os.path.join(project_directory, "ionosphere.csv")
    
    # 1. โหลดข้อมูลแบบไม่มีหัวตาราง (header=None)
    df = pd.read_csv(path_csv, header=None)
    
    # 2. ตั้งชื่อคอลัมน์อัตโนมัติ (นับจำนวนคอลัมน์ทั้งหมด - 1)
    num_features = len(df.columns) - 1
    feature_cols = [f"feature_{i}" for i in range(1, num_features + 1)]
    label_header = "class_label"
    df.columns = feature_cols + [label_header]
    
    # 3. คลีนข้อมูล (แปลง '?' เป็น NaN) และบังคับแปลง Feature ให้เป็นตัวเลข
    df.replace('?', np.nan, inplace=True)
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # 4. เติมค่าว่างด้วย Median ของแต่ละคอลัมน์ (เพื่อไม่ให้สูญเสียแถวข้อมูลแบบ dropna)
    df.fillna(df.median(numeric_only=True), inplace=True)
    
    # 5. บันทึกไฟล์ CSV ที่พร้อมใช้งาน 100%
    df.to_csv(os.path.join(project_directory, "_ionosphere.csv"), index=False)
    
    # --- เข้าสู่กระบวนการโหลดข้อมูลมาตรฐานของระบบ ---
    path_csv = os.path.join(project_directory, "_ionosphere.csv")
    
    df = pd.read_csv(path_csv)
    header_name = df.columns.tolist()
    header_name.remove(label_header)
    
    # ดึงฟังก์ชัน PCA มาตรฐานมาใช้
    csv_pca = csv_to_pca_csv(project_directory, path_csv, label_header)
    header_name_pca = ["PC1", "PC2"]
    
    return path_csv, header_name, label_header, csv_pca, header_name_pca


def load_k_float_encoder_ionosphere(project_directory):
    import numpy as np
    import pandas as pd
    import os
    
    clean_csv = os.path.join(project_directory, "_ionosphere.csv")
    raw_csv = os.path.join(project_directory, "ionosphere.csv")
    
    # 🌟 1. ถ้ายังไม่มีไฟล์คลีน ให้สร้างใหม่จากไฟล์ raw ทันที
    if not os.path.exists(clean_csv):
        df = pd.read_csv(raw_csv, header=None)
        
        # ตั้งชื่อคอลัมน์อัตโนมัติ
        num_features = len(df.columns) - 1
        feature_cols = [f"feature_{i}" for i in range(1, num_features + 1)]
        label_header = "class_label"
        df.columns = feature_cols + [label_header]
        
        # คลีนข้อมูล (แปลง '?' เป็น NaN) และบังคับแปลง Feature ให้เป็นตัวเลข
        df.replace('?', np.nan, inplace=True)
        for col in feature_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # เติมค่าว่างด้วย Median 
        df.fillna(df.median(numeric_only=True), inplace=True)
        
        # บันทึกไฟล์ CSV ที่คลีนแล้ว
        df.to_csv(clean_csv, index=False)

    # 🌟 2. โหลดข้อมูลจากโฟลเดอร์แบบ Dynamic ที่เราเพิ่งทำระบบไว้
    label_name = "class_label"
    result = load_k_float_encoder_to_csv(project_directory, clean_csv, dataset_name="ionosphere")
    
    return result, label_name


def list_folders_in_folder(folder_path):
    folders = [f for f in os.listdir(folder_path) if not os.path.isfile(os.path.join(folder_path, f))]
    return folders


def load_k_float_encoder_to_csv(project_directory, csv_dataset, dataset_name=None):
    result = []
    if dataset_name:
        output_k_fold = os.path.join(project_directory, f"output_k_fold_{dataset_name}")
    else:
        output_k_fold = os.path.join(project_directory, "output_k_fold")
        
    if not os.path.exists(output_k_fold):
        raise FileNotFoundError(f"หาโฟลเดอร์ไม่พบ: {output_k_fold}")

    folders_k = list_folders_in_folder(output_k_fold)
    
    for folder_k in folders_k:
        # ใช้ output_k_fold เป็นฐานที่อยู่ใหม่ (แทนการชี้ไปที่ project_directory แบบเดิม)
        base_fold_path = os.path.join(output_k_fold, folder_k)
        
        train_out = os.path.join(base_fold_path, "training", "data.csv")
        val_out = os.path.join(base_fold_path, "validation", "data.csv")
        test_out = os.path.join(base_fold_path, "test", "data.csv")

        training_files = os.path.join(base_fold_path, "training", "labels")
        validation_files = os.path.join(base_fold_path, "validation", "labels")
        test_files = os.path.join(base_fold_path, "test", "labels")

        # train
        path_csv = os.path.join(project_directory, csv_dataset)
        df = pd.read_csv(path_csv)
        indices = []

        training_files_list = list_files_in_folder(training_files)
        validation_files_list = list_files_in_folder(validation_files)
        test_files_list = list_files_in_folder(test_files)

        for file1 in training_files_list:
            file_name = get_file_name_without_extension(file1)
            indices.append(int(file_name))

        df_train = df.iloc[indices]
        df_train.to_csv(train_out, index=False)

        # val
        df_val = pd.read_csv(path_csv)
        indices = []
        for file1 in validation_files_list:
            file_name = get_file_name_without_extension(file1)
            indices.append(int(file_name))
        df_val = df_val.iloc[indices]
        df_val.to_csv(val_out, index=False)

        # test
        df_test = pd.read_csv(path_csv)
        indices = []
        for file1 in test_files_list:
            file_name = get_file_name_without_extension(file1)
            indices.append(int(file_name))
        df_test = df_test.iloc[indices]
        df_test.to_csv(test_out, index=False)
        
        result.append([train_out, val_out, test_out])

    return result

def load_wine(project_directory):
    path_csv = os.path.join(project_directory, "wine.csv")
    label_header = "Wine"
    df = pd.read_csv(path_csv)
    header_name = df.columns.tolist()
    header_name.remove(label_header)

    # pca
    csv_pca = csv_to_pca_csv(project_directory, path_csv, label_header)
    header_name_pca = ["PC1", "PC2"]
    return path_csv, header_name, label_header, csv_pca, header_name_pca

def load_header_iris(project_directory):
    path_csv = os.path.join(project_directory, "iris.csv")
    label_header = "variety"

    df = pd.read_csv(path_csv)
    header_name = df.columns.tolist()
    header_name.remove(label_header)

    # pca
    csv_pca = csv_to_pca_csv(project_directory, path_csv, label_header)
    header_name_pca = ["PC1", "PC2"]
    return path_csv, header_name, label_header, csv_pca, header_name_pca

def load_header_nutrition(project_directory):
    path_csv = os.path.join(project_directory, "nutrition.csv")
    label_header = "label"

    df = pd.read_csv(path_csv)
    header_name = df.columns.tolist()
    header_name.remove(label_header)

    # pca
    csv_pca = csv_to_pca_csv(project_directory, path_csv, label_header)
    header_name_pca = ["PC1", "PC2"]
    return path_csv, header_name, label_header, csv_pca, header_name_pca

def load_header_student_performance(project_directory):
    path_csv = os.path.join(project_directory, "Student_performance_data.csv")
    label_header = "GradeClass"

    df = pd.read_csv(path_csv)
    header_name = df.columns.tolist()
    header_name.remove(label_header)

    # pca
    csv_pca = csv_to_pca_csv(project_directory, path_csv, label_header)
    header_name_pca = ["PC1", "PC2"]
    return path_csv, header_name, label_header, csv_pca, header_name_pca

def load_header_adult(project_directory):
    print(os.path.join(project_directory, "adult.csv"))
    df = pd.read_csv(os.path.join(project_directory, "adult.csv"))
    # Label Encoding or Ordinal Encoding
    categorical_cols = df.select_dtypes(include=['object']).columns

    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le  # เก็บ encoder ไว้ถ้าต้องการ inverse_transform กลับทีหลัง

    df.to_csv("_adult.csv", index=False)

    # income
    path_csv = os.path.join(project_directory, "_adult.csv")
    label_header = "income"

    df = pd.read_csv(path_csv)
    header_name = df.columns.tolist()
    header_name.remove(label_header)

    # pca
    csv_pca = csv_to_pca_csv(project_directory, path_csv, label_header)
    header_name_pca = ["PC1", "PC2"]
    return path_csv, header_name, label_header, csv_pca, header_name_pca



def load_header_electric(project_directory):
    df = pd.read_csv(os.path.join(project_directory, "electricity.csv"))
    # Label Encoding or Ordinal Encoding
    categorical_cols = df.select_dtypes(include=['object']).columns

    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le  # เก็บ encoder ไว้ถ้าต้องการ inverse_transform กลับทีหลัง

    df.to_csv("_electricity.csv", index=False)

    # income
    path_csv = os.path.join(project_directory, "_electricity.csv")
    label_header = "class"

    df = pd.read_csv(path_csv)
    header_name = df.columns.tolist()
    header_name.remove(label_header)

    # pca
    csv_pca = csv_to_pca_csv(project_directory, path_csv, label_header)
    header_name_pca = ["PC1", "PC2"]
    return path_csv, header_name, label_header, csv_pca, header_name_pca





def load_header_yeast(project_directory): #10 classes unbalance
    path_csv = os.path.join(project_directory, "yeast.csv")
    label_header = "name"

    df = pd.read_csv(path_csv)
    header_name = df.columns.tolist()
    header_name.remove(label_header)

    # pca
    csv_pca = csv_to_pca_csv(project_directory, path_csv, label_header)
    header_name_pca = ["PC1", "PC2"]
    return path_csv, header_name, label_header, csv_pca, header_name_pca

def load_header_churn(project_directory):
    path_csv = os.path.join(project_directory, "Churn.csv")
    label_header = "Churn"

    df = pd.read_csv(path_csv)
    # check_class_imbalance(df, label_header)  #Replace with actual column name
    print("one-hot encode")
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])


    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
    df.dropna(subset=["TotalCharges"], inplace=True)  # Drop rows with NaN
    df = df[df["TotalCharges"] != ""]  # Removes rows where TotalCharges is an empty string
    # print(df.iloc[487])


    encoder = LabelEncoder()
    categorical_columns = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaymentMethod'
    ]

    for col in categorical_columns:
        if col in df.columns:
            df[col] = encoder.fit_transform(df[col])

    print("finished one-hot encode")

    header_name = df.columns.tolist()
    header_name.remove(label_header)

    # non_numeric_columns = df.select_dtypes(exclude=['number']).columns
    # print("Non-numeric columns:", non_numeric_columns.tolist())

    # df.to_csv(os.path.join(project_directory, 'test.csv'), index=False)

    # pca
    csv_pca = csv_to_pca_csv(project_directory, os.path.join(project_directory, 'test.csv'), label_header)
    # csv_pca = df_to_pca_csv(project_directory, df, label_header)
    header_name_pca = ["PC1", "PC2"]
    return path_csv, header_name, label_header, csv_pca, header_name_pca


def check_class_imbalance(df, target_column):
    class_counts = df[target_column].value_counts()
    total_samples = len(df)

    print("Class Distribution:")
    print(class_counts)
    print("\nPercentage Distribution:")
    print((class_counts / total_samples) * 100)

    return class_counts

#https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
def load_header_creditcard(project_directory):
    path_csv = os.path.join(project_directory, "creditcarduci.csv")
    label_header = "default payment next month"

    df = pd.read_csv(path_csv)
    check_class_imbalance(df, label_header)  # Replace with actual column name

    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    # encoder = LabelEncoder()
    # df['Education'] = encoder.fit_transform(df['Education'])
    # df['EmploymentType'] = encoder.fit_transform(df['EmploymentType'])
    # df['MaritalStatus'] = encoder.fit_transform(df['MaritalStatus'])
    # df['HasMortgage'] = encoder.fit_transform(df['HasMortgage'])
    # df['HasDependents'] = encoder.fit_transform(df['HasDependents'])
    # df['LoanPurpose'] = encoder.fit_transform(df['LoanPurpose'])
    # df['HasCoSigner'] = encoder.fit_transform(df['HasCoSigner'])


    header_name = df.columns.tolist()
    header_name.remove(label_header)

    # pca
    csv_pca = csv_to_pca_csv(project_directory, path_csv, label_header)
    header_name_pca = ["PC1", "PC2"]
    return path_csv, header_name, label_header, csv_pca, header_name_pca








def load_abalone(project_directory):
    path_csv = os.path.join(project_directory, "abalone.csv")
    label_header = "Sex"
    df = pd.read_csv(path_csv)
    header_name = df.columns.tolist()
    header_name.remove(label_header)

    return path_csv, header_name, label_header


def load_raisin(project_directory):
    path_csv = os.path.join(project_directory, "Raisin_Dataset.csv")
    label_header = "Class"
    df = pd.read_csv(path_csv)
    header_name = df.columns.tolist()
    header_name.remove(label_header)

    return path_csv, header_name, label_header


def load_header_breast_cancer(project_directory):
    path_csv = os.path.join(project_directory, "breast-cancer-wisconsin1.csv")
    label_header = "diagnosis"
    df = pd.read_csv(path_csv)

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    if "Unnamed: 32" in df.columns:
        df = df.drop(columns=["Unnamed: 32"])

    # df = df.dropna()  # Drop rows that contain any NaN values
    # print(df)

    df.to_csv(os.path.join(project_directory, 'test.csv'), index=False)

    path_csv = os.path.join(project_directory, "test.csv")
    label_header = "diagnosis"

    df = pd.read_csv(path_csv)

    header_name = df.columns.tolist()
    header_name.remove(label_header)


    # pca
    csv_pca = csv_to_pca_csv(project_directory, path_csv, label_header)

    header_name_pca = ["PC1", "PC2"]
    return path_csv, header_name, label_header, csv_pca, header_name_pca

def load_k_float_encoder_breast_cancer(project_directory):
    # อ้างอิงจากไฟล์คลีน (test.csv) ที่ฟังก์ชัน load_header_breast_cancer สร้างไว้
    path_csv = os.path.join(project_directory, "test.csv")
    label_name = "diagnosis"
    
    # 🌟 เรียกโหลดข้อมูลจากโฟลเดอร์แบบ Dynamic (output_k_fold_breast_cancer)
    result = load_k_float_encoder_to_csv(project_directory, path_csv, dataset_name="breast_cancer")
    
    return result, label_name




# https://archive.ics.uci.edu/dataset/10/automobile
def load_header_automobile(project_directory):
    path_csv = os.path.join(project_directory, "automobile.csv")
    df = pd.read_csv(path_csv)

    # จัดการกับข้อมูลที่มีปัญหา (มีเครื่องหมาย ?)
    df.replace('?', np.nan, inplace=True)
    df.dropna(inplace=True)

    # แปลงคอลัมน์ที่เป็นตัวหนังสือให้เป็นตัวเลข (Label Encoding)
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # บันทึกเป็นไฟล์ใหม่ที่คลีนแล้ว
    df.to_csv(os.path.join(project_directory, "_automobile.csv"), index=False)

    # --- เข้าสู่โครงสร้างโหลดข้อมูลมาตรฐาน ---
    path_csv = os.path.join(project_directory, "_automobile.csv")
    label_header = "symboling"
    
    df = pd.read_csv(path_csv)
    header_name = df.columns.tolist()
    header_name.remove(label_header)

    # pca
    csv_pca = csv_to_pca_csv(project_directory, path_csv, label_header)
    header_name_pca = ["PC1", "PC2"]
    
    return path_csv, header_name, label_header, csv_pca, header_name_pca


def load_header_hcc(project_directory):
    import numpy as np
    import pandas as pd
    import os
    
    path_csv = os.path.join(project_directory, "hcc.csv")
    
    # โหลดข้อมูลแบบไม่มีหัวตาราง (header=None)
    df = pd.read_csv(path_csv, header=None)
    
    # ตั้งชื่อคอลัมน์อัตโนมัติให้ตรงตามรูปแบบ ก่อนที่จะแปลงค่า
    num_features = len(df.columns) - 1
    feature_cols = [f"feature_{i}" for i in range(1, num_features + 1)]
    label_header = "class_label"
    df.columns = feature_cols + [label_header]
    
    # จัดการ Missing Value (?) ให้เป็น NaN
    df.replace('?', np.nan, inplace=True)
    
    # บังคับแปลงข้อมูลให้เป็นตัวเลขทั้งหมด
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # 🌟 แก้ปัญหาข้อมูลหาย: แทนที่จะลบทิ้ง (dropna) ให้เติมค่าที่ว่างด้วย "ค่ามัธยฐาน (Median)"
    df.fillna(df.median(), inplace=True) 
    
    # บันทึกไฟล์ CSV ที่คลีนและมีข้อมูลครบ 100%
    df.to_csv(os.path.join(project_directory, "_hcc.csv"), index=False)
    
    # --- เข้าสู่โครงสร้างโหลดข้อมูลมาตรฐาน ---
    path_csv = os.path.join(project_directory, "_hcc.csv")
    label_header = "class_label"
    
    df = pd.read_csv(path_csv)
    header_name = df.columns.tolist()
    header_name.remove(label_header)
    
    # pca 
    csv_pca = csv_to_pca_csv(project_directory, path_csv, label_header)
    header_name_pca = ["PC1", "PC2"]
    
    return path_csv, header_name, label_header, csv_pca, header_name_pca



# lib โครงการผลเฉลยทั้ง 11 ตัว
def load_header_main_category(project_directory, target_label):
    import numpy as np
    import pandas as pd
    import os
    import glob
    
    # 1. ค้นหาไฟล์ต้นฉบับ (รองรับไฟล์ที่ชื่อขึ้นต้นด้วย "export")
    export_files = glob.glob(os.path.join(project_directory, "export*.csv"))
    if not export_files:
        raise FileNotFoundError(f"❌ ไม่พบไฟล์ที่ขึ้นต้นด้วย export... ในโฟลเดอร์ {project_directory}")
    path_csv = export_files[0] # หยิบไฟล์แรกที่เจอมาใช้
    
    # ลองอ่านไฟล์
    try:
        df = pd.read_csv(path_csv, encoding='utf-8-sig')
    except:
        df = pd.read_csv(path_csv)
        
    # 2. แปลงผลเฉลย (Target Label) ให้เป็น 1 และ -1
    df[target_label] = pd.to_numeric(df[target_label], errors='coerce').fillna(-2)
    df[target_label] = np.where(df[target_label] >= 1, 1, -1)
    
    # 3. คัดเฉพาะคอลัมน์ที่เป็นตัวเลข (ข้อความ วันที่ รูปภาพ จะถูกหั่นทิ้งอัตโนมัติ)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_label not in numeric_cols:
        numeric_cols.append(target_label)
    df = df[numeric_cols]
    
    # 4. ป้องกัน Data Leakage: ลบคอลัมน์ผลเฉลยอื่นๆ ทิ้ง ไม่ให้โมเดลโกงข้อสอบ
    other_labels = [col for col in df.columns if col.startswith("main_category___") and col != target_label]
    df = df.drop(columns=other_labels, errors='ignore')
    
    # 5. ตัดคอลัมน์ ID หรือ Metadata ที่ไม่ควรเอามาเทรน
    for drop_col in ['id', '_rootid', '_flag']:
        if drop_col in df.columns:
            df = df.drop(columns=[drop_col])
            
    # 6. เติมค่าว่างด้วย -2 กันระบบพัง (แยกชัดเจนจากค่า 0 เดิม)
    df.fillna(-2, inplace=True)
    
    # 7. ทำ Upsampling (เป็นลำดับสุดท้าย)
    positive_df = df[df[target_label] == 1]
    current_count = len(positive_df)
    
    negative_df = df[df[target_label] == -1]
    neg_count = len(negative_df)
    
    if 0 < current_count < neg_count:
        needed = neg_count - current_count
        sampled_df = positive_df.sample(n=needed, replace=True, random_state=42)
        
        # ----------------------------------------------------
        # 🌟 เพิ่มความ Fuzzy (Synthetic Noise) ให้ข้อมูลที่ก๊อปปี้มา
        # ----------------------------------------------------
        # หาคอลัมน์ที่เป็น Feature ทั้งหมด (ไม่รวมคอลัมน์ผลเฉลย)
        feature_cols = [col for col in sampled_df.columns if col != target_label]
        
        # สร้างค่า Noise เล็กน้อยแบบสุ่ม (Mean=0, Standard Deviation=0.01)
        # ยิ่ง std เยอะ ข้อมูลยิ่งเบี่ยงเบนจากเดิมเยอะ (0.01 = เบี่ยงเบนราวๆ 1%)
        noise = np.random.normal(loc=0.0, scale=0.01, size=sampled_df[feature_cols].shape)
        
        # บวก Noise เข้าไปในข้อมูลที่ถูก Upsample
        sampled_df[feature_cols] = sampled_df[feature_cols] + noise
        # ----------------------------------------------------
        
        df = pd.concat([df, sampled_df], ignore_index=True)
        print(f"📈 [Upsampling + Fuzzy] คลาส {target_label}: เติม Synthetic Data อีก {needed} แถว (เพื่อให้บาลานซ์กับ -1 ที่มี {neg_count} แถว)")
    elif current_count == 0:
        print(f"⚠️ [Upsampling] ข้ามคลาส {target_label}: ไม่มีคลาส 1 อยู่ในข้อมูลเลย")
    
    # บันทึกเป็นไฟล์คลีน แยกตามชื่อโมเดล (เช่น _heritage.csv)
    clean_name = target_label.replace("main_category___", "")
    clean_path = os.path.join(project_directory, f"_{clean_name}.csv")
    df.to_csv(clean_path, index=False, encoding='utf-8-sig')
    
    # โหลดข้อมูลส่งกลับตามโครงสร้างของ Pipeline
    header_name = df.columns.tolist()
    header_name.remove(target_label)

    # คำนวณ pca (คุณต้องมีฟังก์ชัน csv_to_pca_csv เตรียมไว้แล้ว)
    csv_pca = csv_to_pca_csv(project_directory, clean_path, target_label)
    header_name_pca = ["PC1", "PC2"]
    
    return clean_path, header_name, target_label, csv_pca, header_name_pca




def load_header_amphibians_single(project_directory, target_label):
    import pandas as pd
    import numpy as np
    import os
    from sklearn.preprocessing import LabelEncoder

    raw_csv = os.path.join(project_directory, "amphibians.csv")
    
    # ตั้งชื่อไฟล์คลีนตาม label ให้แยกกันชัดเจน ป้องกันการเขียนทับ
    safe_label_name = target_label.replace(' ', '_').replace('-', '_').lower()
    clean_csv = os.path.join(project_directory, f"_amphibians_{safe_label_name}.csv")

    all_label_headers = [
        'Green frogs', 'Brown frogs', 'Common toad',
        'Fire-bellied toad', 'Tree frog', 'Common newt', 'Great crested newt'
    ]

    # 1. ถ้ายังไม่มีไฟล์คลีนสำหรับ Label นี้ ให้สร้างใหม่
    if not os.path.exists(clean_csv):
        df = pd.read_csv(raw_csv, sep=';', skiprows=1)

        # ตัดคอลัมน์ ID ทิ้ง
        if 'ID' in df.columns:
            df = df.drop(columns=['ID'])

        # แปลงข้อมูล Categorical
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

        # *** จุดสำคัญ: Drop Label อีก 6 ตัวที่ไม่ได้ใช้ออก เพื่อไม่ให้ Leak เป็น Feature ***
        labels_to_drop = [lbl for lbl in all_label_headers if lbl != target_label]
        df = df.drop(columns=labels_to_drop, errors='ignore')

        # บันทึกไฟล์ CSV
        df.to_csv(clean_csv, index=False)

    # --- โหลดข้อมูลแบบมาตรฐานเหมือน Dataset อื่นๆ ---
    df_clean = pd.read_csv(clean_csv)
    header_name = df_clean.columns.tolist()
    header_name.remove(target_label)

    csv_pca = csv_to_pca_csv(project_directory, clean_csv, target_label)
    header_name_pca = ["PC1", "PC2"]

    return clean_csv, header_name, target_label, csv_pca, header_name_pca
    


def calculate_confusion_matrix(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("The length of y_true and y_pred must be the same")

    # get class
    classes = np.unique(np.concatenate((y_true, y_pred)))
    num_classes = len(classes)

    # create Confusion Matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)

    # create dictionary
    class_indices = {cls: idx for idx, cls in enumerate(classes)}

    # cal Confusion Matrix
    for true, pred in zip(y_true, y_pred):
        true_idx = class_indices[true]
        pred_idx = class_indices[pred]
        cm[true_idx, pred_idx] += 1

    return cm, classes


def print_confusion_matrix(cm, classes):
    print("Confusion Matrix:")
    print(" " * 10, end="")
    for cls in classes:
        print(f"{cls:10}", end="")
    print()

    for i, cls in enumerate(classes):
        print(f"{cls:10}", end="")
        for j in range(len(classes)):
            print(f"{cm[i, j]:10}", end="")
        print()


def get_file_name_without_extension(file_path):
    # Get the base name (file name with extension)
    base_name = os.path.basename(file_path)

    # Split the base name into the name and extension
    file_name, _ = os.path.splitext(base_name)

    return file_name


def find_majority_element(nums):
    # Count the frequency of each element
    counts = Counter(nums)

    # Find the majority element (if it exists)
    for num, count in counts.items():
        if count > len(nums) // 2:
            return num

    return nums[0]  # No majority element


def is_folder_empty(folder_path):
    # List all items in the directory
    items = os.listdir(folder_path)

    # Check if there's any file in the directory
    for item in items:
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            return False  # The folder has at least one file

    return True  # The folder is empty or contains only subfolders


def clear_folder(path_x):
    if os.path.exists(path_x):
        for filename in os.listdir(path_x):
            file_path = os.path.join(path_x, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Error occurred while deleting {file_path}: {e}")
    else:
        print(f"The path {path_x} does not exist.")


# --data management--
# header_cols = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class_labels']
def read_csv_to_df(file_path, header_cols):
    if not header_cols:
        df = pd.read_csv(file_path)
    else:
        df = pd.read_csv(file_path, names=header_cols)
    return df


def get_last_modified_folder(source_folder):
    all_folders = [f for f in glob.glob(os.path.join(source_folder, '*')) if os.path.isdir(f)]

    if not all_folders:
        return None

    last_modified_folder = max(all_folders, key=os.path.getmtime)
    return last_modified_folder


def list_files_in_folder(folder_path):
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    return files


# np_data_X = features, np_data_Y=result
def split_data_train_validate(np_data_X, np_data_Y, pre_train_validate_indices=None):
    X = np_data_X
    y = np_data_Y

    k = 4  # for train 75% and validate 25%

    # shuffle
    indices = np.arange(len(y))
    np.random.shuffle(indices)

    if pre_train_validate_indices is not None:
        indices = pre_train_validate_indices

    X = X[indices]
    y = y[indices]

    # k folds
    fold_size = len(y) // k
    folds_X = [X[i * fold_size:(i + 1) * fold_size] for i in range(k)]
    folds_y = [y[i * fold_size:(i + 1) * fold_size] for i in range(k)]

    # train test scores
    scores = []

    i = 0

    X_train = np.concatenate([folds_X[j] for j in range(k) if j != i])  # 75%
    y_train = np.concatenate([folds_y[j] for j in range(k) if j != i])  # 75%
    X_validate = folds_X[i]  # 25%
    y_validate = folds_y[i]  # 25%

    return X_train, y_train, X_validate, y_validate, indices


def cross_validation(file_path_csv, pre_indices_test=None, pre_indices_train_validate=None):
    data = pd.read_csv(file_path_csv)
    data['index'] = range(len(data.index))
    data = data[['index'] + [col for col in data.columns if col != 'index']]

    X = data.iloc[:, :-1].values  # features
    y = data.iloc[:, -1].values  # result

    k = 5

    # shuffle
    indices = np.arange(len(y))
    np.random.shuffle(indices)  # random
    if pre_indices_test is not None:
        indices = pre_indices_test
    X = X[indices]
    y = y[indices]

    # k folds
    fold_size = len(y) // k
    folds_X = [X[i * fold_size:(i + 1) * fold_size] for i in range(k)]
    folds_y = [y[i * fold_size:(i + 1) * fold_size] for i in range(k)]

    result_k_fold = []
    for i in range(k):
        X_train = np.concatenate([folds_X[j] for j in range(k) if j != i])
        y_train = np.concatenate([folds_y[j] for j in range(k) if j != i])
        X_test = folds_X[i]
        y_test = folds_y[i]

        X_train, y_train, X_validate, y_validate, indices_train_validate = split_data_train_validate(X_train, y_train, pre_indices_train_validate)
        print('k=', i, 'X_train,y_train=', len(X_train), 'X_validate,y_validate=', len(X_validate), 'X_test,y_test=',
              len(X_test))
        result_k_fold.append([i, X_train, y_train, X_validate, y_validate, X_test, y_test])

    indices_test = indices
    return result_k_fold, indices_test, indices_train_validate


# -----algor.1-----
def preprocess_create_k_fold(file_path_csv, list_analyse_header_name, str_label_header_name,
                             pre_indices_test=None, pre_indices_train_validate=None):


    result_k_fold, indices_test, indices_train_validate = cross_validation(file_path_csv, pre_indices_test, pre_indices_train_validate)
    k_fold_min_features = []
    k_fold_max_features = []

    train_k_fold_index = []
    validate_k_fold_index = []
    test_k_fold_index = []

    i_fold = 0
    for fold in result_k_fold:

        X_train = fold[1]
        X_validate = fold[3]
        X_test = fold[5]

        # index
        first_column = [row[0] for row in X_train]
        first_column_validate = [row[0] for row in X_validate]
        first_column_test = [row[0] for row in X_test]

        train_k_fold_index.append(first_column)
        validate_k_fold_index.append(first_column_validate)
        test_k_fold_index.append(first_column_test)

        i_fold = i_fold + 1

        df_csv = read_csv_to_df(file_path_csv, False)
        df_csv = df_csv[list_analyse_header_name + [str_label_header_name]]
        df_csv[str_label_header_name] = df_csv[str_label_header_name].astype(str)

        # Drop rows not in the list
        df_csv = df_csv.drop(df_csv.index.difference(first_column))

        header_name = list_analyse_header_name
        Ls = []
        Hs = []
        for header in header_name:
            max_value = df_csv[header].max()
            min_value = df_csv[header].min()

            Ls.append(min_value)
            Hs.append(max_value)

        k_fold_min_features.append(Ls)
        k_fold_max_features.append(Hs)

    return (result_k_fold, train_k_fold_index, validate_k_fold_index, test_k_fold_index, k_fold_min_features, k_fold_max_features,
            indices_test, indices_train_validate)


def preprocess_training_strategy(file_path_csv, project_directory, strip_dir, strip_label_dir, strip_label_map_file,
                                 output_k_fold, result_k_fold=None):
    return preprocess_training_strategy_out_path(file_path_csv, project_directory,
                                                 strip_dir,
                                                 strip_label_dir,
                                                 strip_label_map_file,
                                                 output_k_fold, result_k_fold)


def preprocess_training_strategy_out_path(file_path_csv, project_directory,
                                          strip_dir, strip_label_dir, strip_label_map_file,
                                          output_k_fold, result_k_fold):
    output_strip = strip_dir
    output_strip_label = strip_label_dir
    output_strip_label_map = strip_label_map_file

    result_preprocess_training_strategy = []
    result_test_img_folders = []
    result_test_label_folders = []
    if result_k_fold is None:
        print("----use previous cross validation result----")
        result_k_fold = cross_validation(file_path_csv)

    i_fold = 0
    for fold in result_k_fold:
        output_k_fold_index = os.path.join(project_directory, output_k_fold, str(i_fold))
        if not os.path.exists(output_k_fold_index):
            os.makedirs(output_k_fold_index)

        fold_no = fold[0]
        X_train = fold[1]
        y_train = fold[2]
        X_validate = fold[3]
        y_validate = fold[4]
        X_test = fold[5]
        y_test = fold[6]

        # create training
        for X in X_train:
            output_k_fold_index_training = os.path.join(project_directory, output_k_fold, str(i_fold), "training")
            if not os.path.exists(output_k_fold_index_training):
                os.makedirs(output_k_fold_index_training)
            folder_images = os.path.join(project_directory, output_k_fold, str(i_fold), "training", 'images')
            if not os.path.exists(folder_images):
                os.makedirs(folder_images)
            folder_labels = os.path.join(project_directory, output_k_fold, str(i_fold), "training", 'labels')
            if not os.path.exists(folder_labels):
                os.makedirs(folder_labels)

            source_file = os.path.join(output_strip, str(i_fold), str(int(X[0])) + ".jpg")
            path = Path(source_file)
            file_name_with_extension = path.name
            destination_file = os.path.join(folder_images, file_name_with_extension)
            shutil.copy(source_file, destination_file)

            source_file = os.path.join(output_strip_label, str(i_fold), str(int(X[0])) + ".txt")
            path = Path(source_file)
            file_name_with_extension = path.name
            destination_file = os.path.join(folder_labels, file_name_with_extension)
            shutil.copy(source_file, destination_file)

        # create validation
        for X in X_validate:
            output_k_fold_index_validation = os.path.join(project_directory, output_k_fold, str(i_fold), "validation")
            if not os.path.exists(output_k_fold_index_validation):
                os.makedirs(output_k_fold_index_validation)
            folder_images = os.path.join(project_directory, output_k_fold, str(i_fold), "validation", 'images')
            if not os.path.exists(folder_images):
                os.makedirs(folder_images)
            folder_labels = os.path.join(project_directory, output_k_fold, str(i_fold), "validation", 'labels')
            if not os.path.exists(folder_labels):
                os.makedirs(folder_labels)

            source_file = os.path.join(output_strip, str(i_fold), str(int(X[0])) + ".jpg")
            path = Path(source_file)
            file_name_with_extension = path.name
            destination_file = os.path.join(folder_images, file_name_with_extension)
            shutil.copy(source_file, destination_file)

            source_file = os.path.join(output_strip_label, str(i_fold), str(int(X[0])) + ".txt")
            path = Path(source_file)
            file_name_with_extension = path.name
            destination_file = os.path.join(folder_labels, file_name_with_extension)
            shutil.copy(source_file, destination_file)

        # create test
        for X in X_test:
            output_k_fold_index_test = os.path.join(project_directory, output_k_fold, str(i_fold), "test")
            if not os.path.exists(output_k_fold_index_test):
                os.makedirs(output_k_fold_index_test)
            folder_images = os.path.join(project_directory, output_k_fold, str(i_fold), "test", 'images')
            if not os.path.exists(folder_images):
                os.makedirs(folder_images)
            folder_labels = os.path.join(project_directory, output_k_fold, str(i_fold), "test", 'labels')
            if not os.path.exists(folder_labels):
                os.makedirs(folder_labels)

            source_file = os.path.join(output_strip, str(i_fold), str(int(X[0])) + ".jpg")
            path = Path(source_file)
            file_name_with_extension = path.name
            destination_file = os.path.join(folder_images, file_name_with_extension)
            shutil.copy(source_file, destination_file)

            source_file = os.path.join(output_strip_label, str(i_fold), str(int(X[0])) + ".txt")
            path = Path(source_file)
            file_name_with_extension = path.name
            destination_file = os.path.join(folder_labels, file_name_with_extension)
            shutil.copy(source_file, destination_file)

        result_test_img_folders.append(folder_images)
        result_test_label_folders.append(folder_labels)

        # create schema
        # str1 = 'train: ' + os.path.join(project_directory, output_k_fold, str(i_fold), 'training') + '\n'
        # str1 = str1 + 'validation: ' + os.path.join(project_directory, output_k_fold, str(i_fold), 'validation') + "\n"
        # str1 = str1 + "\n"
        #
        # with open(output_strip_label_map, 'r') as file:
        #     content = file.read()
        # list_from_string = content.split(',')
        # str1 = str1 + "nc: " + str(len(list_from_string)) + "\n"
        # str1 = str1 + "\n"
        #
        # str1 = str1 + "name: " + str(content) + "\n"
        #
        # yaml_file = os.path.join(project_directory, output_k_fold, str(i_fold), 'dataset.yaml')

        # create schema (dataset.yaml)  --- FIX ---
        train_dir = os.path.join(project_directory, output_k_fold, str(i_fold), "training")
        val_dir = os.path.join(project_directory, output_k_fold, str(i_fold), "validation")
        test_dir = os.path.join(project_directory, output_k_fold, str(i_fold), "test")

        with open(output_strip_label_map, "r") as file:
            content = file.read()

        # content เดิมเป็น "['1','2','3']" อยู่แล้ว
        str1 = f"train: {train_dir}\n"
        str1 += f"val: {val_dir}\n"
        str1 += f"test: {test_dir}\n\n"
        str1 += f"nc: {len(content.split(','))}\n"
        str1 += f"names: {content}\n"

        yaml_file = os.path.join(project_directory, output_k_fold, str(i_fold), "dataset.yaml")
        with open(yaml_file, "w", encoding="utf-8") as f:
            f.write(str1)

        if os.path.exists(yaml_file):
            os.remove(yaml_file)
        with open(yaml_file, 'a') as file:
            file.write(str1)

        result_preprocess_training_strategy.append(yaml_file)

        # generate bat file to train model
        model_train_output = os.path.join(project_directory, "runs", "train")
        if os.path.exists(model_train_output):
            shutil.rmtree(model_train_output)
        os.makedirs(model_train_output)

        last_folder = os.path.join(model_train_output)
        # file_weight = os.path.join(last_folder, "exp", "weights", "best.pt")
        train_py = os.path.join(project_directory, "train.py")
        yolo5s = os.path.join(project_directory, "yolov5s.pt")

        training_bat_strx = "python " + train_py + " --data " + yaml_file + \
                           " --weights " + yolo5s + " --img 640 --batch-size 2 --epochs 1000"

        training_bat_str = "START " + training_bat_strx

        training_bat_path = os.path.join(project_directory, output_k_fold, str(i_fold),
                                         'train.bat')
        with open(training_bat_path, 'w') as file:
            file.write(training_bat_str)

        print("to train model run (CMD) > \n" + training_bat_strx)
        print("OR to train model run the train.py at: " + training_bat_path)
        i_fold = i_fold + 1

    return result_preprocess_training_strategy, result_test_img_folders, result_test_label_folders, result_k_fold


# --encoder transfer--
def strips_encode(csv_file, list_analyse_header_name, str_label_header_name, output_strip_label_map,
                  min_feature_k_fold, max_feature_k_fold, output_strip, output_strip_label):
    df_csv = read_csv_to_df(csv_file, False)
    df_csv = df_csv[list_analyse_header_name + [str_label_header_name]]
    df_csv[str_label_header_name] = df_csv[str_label_header_name].astype(str)

    header_name = list_analyse_header_name
    label = str_label_header_name

    df_labels = df_csv.groupby(label)
    labels = []
    for name, group in df_labels:
        labels.append(str(name))

    if os.path.exists(output_strip_label_map):
        os.remove(output_strip_label_map)
    with open(output_strip_label_map, 'a') as file:
        file.write(str(labels))

    Ls = min_feature_k_fold
    Hs = max_feature_k_fold

    output_strip_fold = os.path.join(output_strip)
    if not os.path.exists(output_strip_fold):
        os.makedirs(output_strip_fold)

    output_strip_label_fold = os.path.join(output_strip_label)
    if not os.path.exists(output_strip_label_fold):
        os.makedirs(output_strip_label_fold)

    for index, row in df_csv.iterrows():
        Values = []
        for index_header, header in enumerate(header_name):
            Values.append(row[header])


        img_strip1 = strip_encode(Values, Ls, Hs)
        img_strip1.save(os.path.join(output_strip_fold, str(index) + '.jpg'))

        label_index = labels.index(str(row[label]))

        with open(os.path.join(output_strip_label_fold, str(index) + '.txt'), 'a') as file:
            file.write(str(label_index) + " 0.5 0.5 1 1")  # 0 0 1 1 is whole image detection

    print("output location: ")
    print(output_strip)
    print(output_strip_label)
    print(output_strip_label_map)


# --strip enchoder--
# solution scope inputs are value + or -(float or int)

def strip_encode(Values, Ls, Hs):
    return strip_encode1(Values, Ls, Hs)

#======encoder v1=======
# Values=list of data values (+-float or +-int), L is min bound, H is max bound,
# each_strip_size = square box size of each value
# max_col_strip_item = max image strip concatinate on width
def strip_encode1(Values, Ls, Hs, color_intensity=255, img_size_for_each=10):
    strip_images = []
    # generate each strip
    index = 0
    for value in Values:

        L = Ls[index]
        H = Hs[index]

        if value < L:
            value = L
        if value > H:
            value = H

        if (H == L):  # no difference value set to 255
            interpolated_value = 255
        else:

            interpolated_value = (((value - L) / (H - L)) * color_intensity)
            # print("value=",value,"L=", L,"H=", H,"interpolated_value=", interpolated_value)
            # exit(0)
        # print(value, Ls[index], Hs[index],interpolated_value)
        img = generate_strip(interpolated_value, img_size_for_each)
        # img = insert_boarder_to_img(img)

        # insert text value to image
        if (interpolated_value > 200):
            insert_text_to_img(img, str(value), (0, 0, 0))
        else:
            insert_text_to_img(img, str(value))
        strip_images.append(img)
        index = index + 1

    img_concate = concatinate_list_img_left2right(strip_images)


    img_fill_emty = generate_strip(0, img_size_for_each)
    # img_fill_emty = insert_boarder_to_img(img_fill_emty)
    matrix_imgs = list_to_square_matrix(strip_images, img_fill_emty)

    result_lines_img = []
    for line_img in matrix_imgs:
        result_lines_img.append(concatinate_list_img_left2right(line_img))

    img_concate = concatinate_list_img_top2bot(result_lines_img)

    return img_concate


def covert_array_to_plt(image_arrays):
    # Convert arrays to PIL Images
    images = [Image.fromarray(image_array) for image_array in image_arrays]
    return images


# interpolated_value 0-255
def generate_strip(interpolated_value, each_strip_size):
    strip_image = create_blank_image(each_strip_size, each_strip_size, interpolated_value)
    return strip_image


def generate_strip_color(R, G, B, width, height):
    image = Image.new("RGB", (width, height))

    pixels = image.load()

    for x in range(width):
        for y in range(height):
            r = R
            g = G
            b = B
            pixels[x, y] = (r, g, b)

    return image


def create_blank_image(width, height, intensity):
    blank_image = np.full((height, width, 3), intensity, dtype=np.uint8)
    blank_image = Image.fromarray(blank_image)
    return blank_image


def concatinate_strip_top2bot(image1, image2):
    if image1.width != image2.width:
        raise ValueError("The widths of the images are not the same.")

    combined_height = image1.height + image2.height
    combined_image = Image.new('RGB', (image1.width, combined_height))

    combined_image.paste(image1, (0, 0))
    combined_image.paste(image2, (0, image1.height))

    return combined_image


def concatinate_list_img_top2bot(img_list):
    img_concate = img_list[0]
    count = 0
    for img_line in img_list:
        if (count != 0):
            img_concate = concatinate_strip_top2bot(img_concate, img_line)

        count = count + 1

    return img_concate

def concatinate_strip_left2right(image1, image2):
    # Check if the heights are the same
    if image1.height != image2.height:
        raise ValueError("The heights of the images are not the same.")

    # Create a new image with the combined width
    combined_width = image1.width + image2.width
    combined_image = Image.new('RGB', (combined_width, image1.height))

    # Paste the images into the combined image
    combined_image.paste(image1, (0, 0))
    combined_image.paste(image2, (image1.width, 0))

    return combined_image

def concatinate_list_img_left2right(img_list):
    img_concate = img_list[0]
    count = 0
    for img_line in img_list:
        if (count != 0):
            img_concate = concatinate_strip_left2right(img_concate, img_line)

        count = count + 1

    return img_concate


def concatinate_strip_line(image1, image2):
    # concatinate image from top to bot
    if image1.shape[1] != image2.shape[1]:
        raise ValueError("Images must have the same width for vertical concatenation.")

    concatenated_image = cv2.vconcat([image1, image2])
    return concatenated_image


# -----algor.2-----

def preprocess_training_strategy2(file_path_csv, project_directory, result_k_fold=None):
    out_strip = "out_strip2"
    out_strip_label = "out_strip_label2"
    output_strip_label_map = "output_strip_label_map"
    output_k_fold = "output_k_fold2"

    return preprocess_training_strategy_out_path(file_path_csv, project_directory,
                                                 out_strip,
                                                 out_strip_label,
                                                 output_strip_label_map,
                                                 output_k_fold, result_k_fold)

def make_square_img(image_path, output_path):
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # find padding sizes
    if height > width:
        pad_size = (height - width) // 2
        padding = ((0, 0), (pad_size, pad_size), (0, 0))  # (top-bottom, left-right, color channels)
    else:
        pad_size = (width - height) // 2
        padding = ((pad_size, pad_size), (0, 0), (0, 0))  # (top-bottom, left-right, color channels)

    # add padding to make the image square <<--
    square_image = np.pad(image, padding, mode='constant', constant_values=0)  # 0 for black padding

    cv2.imwrite(output_path, square_image)



# "123456789" --> ['1', '23', '45', '67', '89']
def split_into_pairs(s):
    # Start with first element if the string length is odd
    result = [s[0]] if len(s) % 2 != 0 else []
    # Then add pairs of 2 characters
    result += [s[i:i + 2] for i in range(len(s) % 2, len(s), 2)]
    return result


def split_into_pairs_left_to_right(lst):
    # Split into pairs
    return [lst[i:i + 2] for i in range(0, len(lst), 2)]


def create_feature_for_strip_color(x, digit_block=3):
    Values = []

    #eliminate significant
    data = x.split('.')
    digits = split_into_pairs(str(abs(int(data[0]))))
    digits = list(map(int, digits))  # str --> int

    if (len(digits) > digit_block):  # pad 0
        print("your feature value is bigger than +999,999", data)
        exit(0)

    digit_block_length = len(digits)
    for i in range(digit_block - digit_block_length):
        digits.insert(0, 0)

    for d in digits:
        if int(data[0]) < 0: # negativa digit
            Values.append(-1 * d)
        else: # positive digit
            Values.append(d)

    return Values

def interpolate_value(L, H, value):
    if (H != L):

        if (value < L):
            value = L
        if (value > H):
            value = H

        interpolated_value = ((value - L) * 240 / (H - L)) + 10

    else:  # no difference value set to 0
        interpolated_value = 0

    return interpolated_value

def safe_value_for_strip(x, limit=999999):
    """แปลงค่าให้เป็น float และ clamp ให้อยู่ในช่วง [-limit, +limit]"""
    try:
        val = float(x)
    except:
        return 0.0
    if val > limit:
        return limit
    elif val < -limit:
        return -limit
    return val


# ====== encoder v2 (patched with normalization & safe clamp) ======
def strips_encode2(csv_file, list_analyse_header_name, str_label_header_name,
                   output_strip_label_map, min_feature_k_fold, max_feature_k_fold,
                   output_strip, output_strip_label, img_size_for_each=10,
                   norm_limit=999_999):
    import shutil, os

    # --- เตรียมโฟลเดอร์ใหม่ ---
    if os.path.exists(output_strip):
        shutil.rmtree(output_strip)
    os.makedirs(output_strip)

    if os.path.exists(output_strip_label):
        shutil.rmtree(output_strip_label)
    os.makedirs(output_strip_label)

    # --- โหลด CSV ---
    df_csv = read_csv_to_df(csv_file, False)
    df_csv[str_label_header_name] = df_csv[str_label_header_name].astype(str)

    # --- Normalize ทุก feature ให้อยู่ในช่วง [-norm_limit, +norm_limit] ---
    min_features1 = min_feature_k_fold.copy()
    max_features1 = max_feature_k_fold.copy()

    for i, header in enumerate(list_analyse_header_name):
        L = float(min_features1[i])
        H = float(max_features1[i])
        if H == L:
            df_csv[header] = 0.0
            continue

        # Normalize → [-1, +1]
        vals = df_csv[header].astype(float)
        norm_vals = 2 * ((vals - L) / (H - L)) - 1
        norm_vals = norm_vals.clip(-1.0, 1.0)
        # Scale → [-norm_limit, +norm_limit]
        df_csv[header] = (norm_vals * norm_limit).round()

    # --- เตรียม label classes ---
    label = str_label_header_name
    labels = [str(name) for name, _ in df_csv.groupby(label)]
    label_rows = [row[label] for _, row in df_csv.iterrows()]

    # --- Encode features เป็น strip (มี clamp) ---
    rows_f_new = []
    for _, row in df_csv.iterrows():
        f_news = []
        for header in list_analyse_header_name:
            x_val = safe_value_for_strip(row[header], limit=norm_limit)
            values = create_feature_for_strip_color(str(int(x_val)))
            f_news.append(values)
        rows_f_new.append(f_news)

    # --- คำนวณ min/max หลัง normalize (ช่วง -norm_limit → +norm_limit) ---
    new_min_features = [create_feature_for_strip_color(str(-norm_limit)) for _ in list_analyse_header_name]
    new_max_features = [create_feature_for_strip_color(str(norm_limit)) for _ in list_analyse_header_name]

    # --- วาดภาพ + label ---
    row_num = 0
    for f_news in rows_f_new:
        strip_img = []
        for f_num, Values in enumerate(f_news):
            Ls = new_min_features[f_num]
            Hs = new_max_features[f_num]

            # ใช้ค่า RGB สามช่อง (เดิมใช้ Hs[1] ซ้ำ → แก้ให้ถูกต้อง)
            d0 = interpolate_value(Ls[0], Hs[0], Values[0])
            d1 = interpolate_value(Ls[1], Hs[1], Values[1])
            d2 = interpolate_value(Ls[2], Hs[2], Values[2])

            img = generate_strip_color(round(d0), round(d1), round(d2),
                                       img_size_for_each, img_size_for_each)

            if d2 > 100:
                insert_text_to_img(img, str(Values[2]), (0, 0, 0))
            else:
                insert_text_to_img(img, str(Values[2]))

            strip_img.append(img)

        # ทำให้เป็นสี่เหลี่ยมจัตุรัสพอดี
        img_fill_empty = generate_strip_color(255, 255, 255,
                                              img_size_for_each, img_size_for_each)
        matrix_imgs = list_to_square_matrix(strip_img, img_fill_empty)
        result_lines_img = [concatinate_list_img_left2right(line_img) for line_img in matrix_imgs]
        img_concate = concatinate_list_img_top2bot(result_lines_img)

        img_concate.save(os.path.join(output_strip, f"{row_num}.jpg"))

        label_index = labels.index(label_rows[row_num])
        with open(os.path.join(output_strip_label, f"{row_num}.txt"), 'a') as file:
            file.write(f"{label_index} 0.5 0.5 1 1")

        row_num += 1

    print("output location:")
    print(output_strip)
    print(output_strip_label)
    print(output_strip_label_map)


def insert_boarder_to_img(img_plt):
    border_size = 1
    return ImageOps.expand(img_plt, border=border_size, fill='black')

def insert_text_to_img(img_plt, insert_text, color=(255, 255, 255)):
    if _insert_text_to_img_mode == 0:
        img_plt
    else:
        # Load the image
        image = img_plt

        # Initialize ImageDraw
        draw = ImageDraw.Draw(image)

        # Define the text to insert
        text = insert_text

        try:
            font = ImageFont.truetype("arial.ttf", 7)
        except IOError:
            # กรณีไม่มีฟอนต์ "arial.ttf" จะใช้ฟอนต์เริ่มต้น
            font = ImageFont.load_default()

        # Choose a font and size (you can download fonts if needed)
        # If you don't specify a font, Pillow will use a default one.

        # Define the position (x, y) where you want to place the text
        position = (0, 0)

        # Add the text to the image
        draw.text(position, text, font=font, fill=color)

        return image

def max_min_range_expand(min_list, max_list):
    b = max_list
    a = min_list
    range_of_max_min = [(b - a) for b, a in zip(b, a)]
    a_new = [(a - range_of_max_min) for a, range_of_max_min in zip(a, range_of_max_min)]
    b_new = [(b + range_of_max_min) for b, range_of_max_min in zip(b, range_of_max_min)]
    return a_new, b_new


def list_to_square_matrix(lst, fill_left_with=0):
    n = math.ceil(math.sqrt(len(lst)))
    lst.extend([fill_left_with] * (n * n - len(lst)))

    matrix = []
    for i in range(n):
        row = lst[i * n:(i + 1) * n]
        matrix.append(row)

    return matrix

def list_to_regtangle(lst, fill_left_with=0):
    x = math.ceil(math.sqrt(len(lst)))*2
    # Height is half the width
    height = x // 2
    # Width is 2 times the height
    width = 2 * height
    # Create a matrix (list of lists) filled with zeros
    matrix = [[fill_left_with] * width for _ in range(height)]

    value_index = 0
    for i in range(height):
        for j in range(width):
            if value_index < len(lst):
                matrix[i][j] = lst[value_index]
                value_index += 1

    return matrix



def create_feature_digit(x, digit_block=3):
    Values = []

    if "e" in x or "E" in x:
        x = f"{float(x):.20f}"  # Convert to full decimal

    data1 = x.split('-')
    sign = "+"
    if len(data1) == 2:
        sign = "-"
        data = data1[1]
    else:
        data = data1[0]
    data = data.split('.')

    # integer
    digits = split_into_pairs(str(abs(int(data[0]))))

    digits = list(map(int, digits))  # str --> int

    if (len(digits) > digit_block):  # pad 0
        print("your feature value is bigger than 999,999 to -999,999")
        exit(0)

    digit_block_length = len(digits)
    for i in range(digit_block - digit_block_length):
        digits.insert(0, 0)

    for d in digits:
        if sign == '-':  # negativa digit
            Values.append(-1 * d)
        else:  # positive digit
            Values.append(d)

    # significant
    if(len(data) == 2):
        significant = str(abs(int(data[1])))
        if(len(significant) % 2 != 0):
            significant = significant + '0'

        significants = split_into_pairs_left_to_right(significant)
        digit_block_length = len(significants)
        for i in range(digit_block - digit_block_length):
            significants.append('00')
        significants = significants[:digit_block]
    else:
        significants = []
        digit_block_length = len(significants)
        for i in range(digit_block - digit_block_length):
            significants.append('00')

    significants = list(map(int, significants))

    for d in significants:
        if sign == '-':  # negativa digit
            Values.append(-1 * d)
        else:  # positive digit
            Values.append(d)

    return Values

#======encoder v3.1=======
# def strips_encode3(csv_file, list_analyse_header_name, str_label_header_name,
#                    output_strip_label_map, min_feature_k_fold, max_feature_k_fold,
#                    output_strip, output_strip_label, digit_block=3, img_size_for_each=10):
#     min_features1 = min_feature_k_fold.copy()
#     max_features1 = max_feature_k_fold.copy()
#
#     if os.path.exists(output_strip):
#         shutil.rmtree(output_strip)
#     os.makedirs(output_strip)
#
#     if os.path.exists(output_strip_label):
#         shutil.rmtree(output_strip_label)
#     os.makedirs(output_strip_label)
#
#     # --adjust n significant--
#     df_csv = read_csv_to_df(csv_file, False)
#     df_csv[str_label_header_name] = df_csv[str_label_header_name].astype(str)
#
#     # get class from colume label
#     label = str_label_header_name
#     df_labels = df_csv.groupby(label)
#     labels = []  # all classes unique possible value
#     for name, group in df_labels:
#         labels.append(str(name))
#
#     # keep label of each row
#     label_rows = []
#     for index, row in df_csv.iterrows():  # (each row)
#         label_rows.append(row[label])
#
#     # eliminate significant
#     rows_f_new = []
#     for index, row in df_csv.iterrows():  # (each row)
#         f_news = []
#         # single feature
#         for index_header, header in enumerate(list_analyse_header_name):
#             x = str(row[header])
#             values = create_feature_digit(x)
#
#             f_news.append(values)
#
#         rows_f_new.append(f_news)
#
#     new_min_features = []
#     for feature in min_features1:
#         # print(feature)s
#         values = create_feature_digit(str(feature))
#         new_min_features.append(values)
#
#     new_max_features = []
#     for feature in max_features1:
#         values = create_feature_digit(str(feature))
#         new_max_features.append(values)
#
#     row_num = 0
#     for f_news in rows_f_new:
#
#         f_num = 0
#         strip_img = []
#         for Values in f_news:
#             Ls = new_min_features[f_num]
#             Hs = new_max_features[f_num]
#
#             imgs = []
#             for i in range(digit_block*2):
#
#                 d0 = interpolate_value(Ls[i], Hs[i], Values[i])
#                 img0 = generate_strip(d0, img_size_for_each)
#
#                 # insert text value to image (this line, not involve with algorithm 3)
#                 if (d0 > 100):
#                     insert_text_to_img(img0, str(Values[i]), (0, 0, 0))
#                 else:
#                     insert_text_to_img(img0, str(Values[i]))
#
#                 imgs.append(img0)
#
#             img_digit_feature = concatinate_list_img_left2right(imgs)
#
#             strip_img.append(img_digit_feature)
#             f_num = f_num + 1
#
#         img_concate = concatinate_list_img_top2bot(strip_img)
#         img_concate.save(os.path.join(output_strip, str(row_num) + '.jpg'))
#         make_square_img(os.path.join(output_strip, str(row_num) + '.jpg'), os.path.join(output_strip, str(row_num) + '.jpg'))
#
#         label_index = labels.index(label_rows[row_num])
#         with open(os.path.join(output_strip_label, str(row_num) + '.txt'), 'a') as file:
#             file.write(str(label_index) + " 0.5 0.5 1 1")
#
#         row_num = row_num + 1
#
#     print("output location: ")
#     print(output_strip)
#     print(output_strip_label)
#     print(output_strip_label_map)

#======encoder v3.2=======
def strips_encode3(csv_file, list_analyse_header_name, str_label_header_name,
                   output_strip_label_map, min_feature_k_fold, max_feature_k_fold,
                   output_strip, output_strip_label, digit_block=3, img_size_for_each=10,
                   norm_limit=999_999):

    min_features1 = min_feature_k_fold.copy()
    max_features1 = max_feature_k_fold.copy()

    if os.path.exists(output_strip):
        shutil.rmtree(output_strip)
    os.makedirs(output_strip)

    if os.path.exists(output_strip_label):
        shutil.rmtree(output_strip_label)
    os.makedirs(output_strip_label)

    df_csv = read_csv_to_df(csv_file, False)
    df_csv[str_label_header_name] = df_csv[str_label_header_name].astype(str)

    # === NEW: normalize ทุก feature เป็น ±norm_limit ก่อน ===
    for i, header in enumerate(list_analyse_header_name):
        L = float(min_features1[i]); H = float(max_features1[i])
        if H == L:
            df_csv[header] = 0.0
            continue
        vals = df_csv[header].astype(float)
        norm_vals = 2 * ((vals - L) / (H - L)) - 1            # [-1, +1]
        norm_vals = norm_vals.clip(-1.0, 1.0)
        df_csv[header] = (norm_vals * norm_limit).round()     # [-norm_limit, +norm_limit]

    # class mapping
    label = str_label_header_name
    labels = [str(name) for name, _ in df_csv.groupby(label)]
    label_rows = [row[label] for _, row in df_csv.iterrows()]

    # encode เป็น digit หลัง normalize (จะไม่เกิน ±999,999 แล้ว)
    rows_f_new = []
    for _, row in df_csv.iterrows():
        f_news = []
        for header in list_analyse_header_name:
            x_val = safe_value_for_strip(row[header], limit=norm_limit)
            values = create_feature_digit(str(int(x_val)), digit_block=digit_block)
            f_news.append(values)
        rows_f_new.append(f_news)

    # min/max ใหม่หลัง normalize (ใช้ขอบคงที่)
    new_min_features = [create_feature_digit(str(-norm_limit), digit_block=digit_block)
                        for _ in list_analyse_header_name]
    new_max_features = [create_feature_digit(str( norm_limit), digit_block=digit_block)
                        for _ in list_analyse_header_name]

    row_num = 0
    for f_news in rows_f_new:
        strip_img = []
        for f_num, Values in enumerate(f_news):
            Ls = new_min_features[f_num]
            Hs = new_max_features[f_num]

            imgs = []
            for i in range(digit_block * 2):
                d0 = interpolate_value(Ls[i], Hs[i], Values[i])
                img0 = generate_strip(d0, img_size_for_each)

                if d0 > 100:
                    insert_text_to_img(img0, str(Values[i]), (0, 0, 0))
                else:
                    insert_text_to_img(img0, str(Values[i]))

                imgs.append(img0)

            img_digit_feature = concatinate_list_img_left2right(imgs)
            strip_img.append(img_digit_feature)

        img_concate = concatinate_list_img_top2bot(strip_img)
        img_concate.save(os.path.join(output_strip, f"{row_num}.jpg"))
        make_square_img(os.path.join(output_strip, f"{row_num}.jpg"),
                        os.path.join(output_strip, f"{row_num}.jpg"))

        label_index = labels.index(label_rows[row_num])
        with open(os.path.join(output_strip_label, f"{row_num}.txt"), 'a') as file:
            file.write(f"{label_index} 0.5 0.5 1 1")

        row_num += 1

    print("output location: ")
    print(output_strip)
    print(output_strip_label)
    print(output_strip_label_map)


def convert_np_to_list(obj):
    """
    Recursively check if the element is a NumPy array or list and convert it to a list.
    """
    if isinstance(obj, np.ndarray):
        # Convert NumPy array to a list
        return obj.tolist()
    elif isinstance(obj, list):
        # Recursively apply the function to each element if it's a list
        return [convert_np_to_list(item) for item in obj]
    else:
        # If it's neither a NumPy array nor a list, return the object itself
        return obj

def convert_np_types(obj):
    """
    Recursively convert NumPy types to native Python types.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert NumPy arrays to lists
    elif isinstance(obj, list):
        return [convert_np_types(item) for item in obj]  # Recursively handle lists
    else:
        return obj  # Return other types as-is