# strip_pre2.py
#
# Example:
#   python strip_pre2.py --datasets wine --cell auto --dataset_root "C:\yolo8_strip\ultralytics"
#   python strip_pre2.py --datasets iris --cell auto --dataset_root "C:\yolo8_strip\ultralytics"
#   python strip_pre2.py --datasets automobile --cell auto --dataset_root "C:\yolo8_strip\ultralytics"
#   python strip_pre2.py --datasets breast_cancer --cell auto --dataset_root "C:\yolo8_strip\ultralytics"
#   python strip_pre2.py --datasets ionosphere --cell auto --dataset_root "C:\yolo8_strip\ultralytics"
#
# Note:
# - This script expects dataset CSV to exist at: <dataset_root>\<dataset>\<expected_filename>
#   (we will ensure it in run_all_datasets.py by copying from dataset_root)

# strip_pre2.py
import lib
import os
import sys
import json
import subprocess
import argparse
from pathlib import Path

CATEGORY_MAP = {
    "heritage": "main_category___heritage_ecosystem",
    "way_of_life": "main_category___way_of_life",
    "narrative_symbol": "main_category___narrative_symbol",
    "tradition_culture": "main_category___tradition_culture",
    "food_spa_herbs": "main_category___food_spa_herbs",
    "values_belief_wisdom": "main_category___values_belief_wisdom",
    "crafts_community_products": "main_category___crafts_community_products",
    "symbols_indigenous": "main_category___symbols_indigenous",
    "culture_architecture": "main_category___culture_architecture",
    "traditional_resources": "main_category___traditional_resources",
    "craft_knowledge": "main_category___craft_knowledge"
}

AMPHIBIANS_MAP = {
    "amphibians_green_frogs": "Green frogs",
    "amphibians_brown_frogs": "Brown frogs",
    "amphibians_common_toad": "Common toad",
    "amphibians_fire_bellied_toad": "Fire-bellied toad",
    "amphibians_tree_frog": "Tree frog",
    "amphibians_common_newt": "Common newt",
    "amphibians_great_crested_newt": "Great crested newt"
}

DATASET_ALIASES = {
    "wine": "wine",
    "iris": "iris",
    "breast_cancer": "breast_cancer",
    "student": "student",
    "adult": "adult",
    "electric": "electric",
    "automobile": "automobile",
    "hcc": "hcc",
    "ionosphere":"ionosphere",
    
}

for k in CATEGORY_MAP.keys():
    DATASET_ALIASES[k] = k

for k in AMPHIBIANS_MAP.keys():
    DATASET_ALIASES[k] = k

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def select_dataset(dataset_name: str, output_dir: str):
    if dataset_name == "wine":
        return lib.load_wine(output_dir)
    if dataset_name == "iris":
        return lib.load_header_iris(output_dir)
    if dataset_name == "breast_cancer":
        return lib.load_header_breast_cancer(output_dir)
    if dataset_name == "student":
        return lib.load_header_student_performance(output_dir)
    if dataset_name == "adult":
        return lib.load_header_adult(output_dir)
    if dataset_name == "electric":
        return lib.load_header_electric(output_dir)
    if dataset_name == "automobile":                       
        return lib.load_header_automobile(output_dir)
    if dataset_name == "hcc":                       
        return lib.load_header_hcc(output_dir)
    if dataset_name == "ionosphere":
        return lib.load_header_ionosphere(output_dir)

    
    # 11ตัวโครงการ
    if dataset_name in CATEGORY_MAP:
        target_label = CATEGORY_MAP[dataset_name]
        return lib.load_header_main_category(output_dir, target_label)

    # Amphibians 7 โมเดล
    if dataset_name in AMPHIBIANS_MAP:
        target_label = AMPHIBIANS_MAP[dataset_name]
        return lib.load_header_amphibians_single(output_dir, target_label)

    raise ValueError(f"Unknown dataset: {dataset_name}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--cell", default="auto")
    ap.add_argument("--dataset_root", default=r"C:\yolo8_strip\ultralytics")
    ap.add_argument("--k_list", default="0", help="comma-separated K values e.g. 0,2,4,8,16")
    args = ap.parse_args()

    dataset_name = args.datasets.strip()
    if dataset_name not in DATASET_ALIASES:
        raise SystemExit(f" Unknown dataset '{dataset_name}'. Allowed: {list(DATASET_ALIASES.keys())}")

    k_values = [int(x.strip()) for x in args.k_list.split(",") if x.strip() != ""]

    script_dir = Path(__file__).parent.resolve()
    dataset_root = Path(args.dataset_root).resolve()
    dataset_out_dir = dataset_root / dataset_name
    ensure_dir(dataset_out_dir)

    print("\n====================================================")
    print(f"DATASET: {dataset_name}")
    print("====================================================\n")

    _output_directory = str(script_dir)
    
    _output_label_list = os.path.join(_output_directory, f"output_strip_label_map_{dataset_name}.txt")
    _output_kfold_data = os.path.join(_output_directory, f"output_kfold_data_{dataset_name}.json")

    _output_strip_encode_v1 = os.path.join(_output_directory, f"out_strip_{dataset_name}")
    _output_strip_encode_label_v1 = os.path.join(_output_directory, f"out_strip_label_{dataset_name}")
    _output_k_fold_v1 = os.path.join(_output_directory, f"output_k_fold_{dataset_name}")
    
    _output_strip_encode_v2 = os.path.join(_output_directory, f"out_strip2_{dataset_name}")
    _output_strip_encode_label_v2 = os.path.join(_output_directory, f"out_strip_label2_{dataset_name}")
    _output_k_fold_v2 = os.path.join(_output_directory, f"output_k_fold2_{dataset_name}")
    
    _output_strip_encode_v3 = os.path.join(_output_directory, f"out_strip3_{dataset_name}")
    _output_strip_encode_label_v3 = os.path.join(_output_directory, f"out_strip_label3_{dataset_name}")
    _output_k_fold_v3 = os.path.join(_output_directory, f"output_k_fold3_{dataset_name}")
    
    folders = [
        _output_strip_encode_v1, _output_strip_encode_label_v1, _output_k_fold_v1,
        _output_strip_encode_v2, _output_strip_encode_label_v2, _output_k_fold_v2,
        _output_strip_encode_v3, _output_strip_encode_label_v3, _output_k_fold_v3,
    ]

    # นี่คือบรรทัดที่มันดึงไฟล์ CSV ออกมาจากระบบของคุณ
    _path_csv_raw, _header_name_raw, _label_header, _pca_csv, header_name_pca = select_dataset(
        dataset_name, str(dataset_out_dir)
    )

    # ================ โค้ดดึง Min-Max ที่ผ่านการ Expand แบบถูกต้อง 100% ================
    print(f"\n🎯 [min-max] กำลังดึงค่า Min-Max จากไฟล์ต้นขั้ว: {_path_csv_raw}")
    try:
        import pandas as pd
        df = pd.read_csv(_path_csv_raw)

        print(df.columns)
        
        # ค้นหาและตัดคอลัมน์ผลลัพธ์ (Label) ออก
        if isinstance(_label_header, list):
            # กรณี Multi-label (_label_header เป็น List) ให้ตรวจสอบและ Drop ทีละคอลัมน์
            cols_to_drop = [col for col in _label_header if col in df.columns]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
        else:
            # กรณี Single-label (_label_header เป็น String)
            label_col = _label_header if _label_header in df.columns else "class_label"
            if label_col in df.columns:
                df = df.drop(columns=[label_col])
            
        # 1. คัดเฉพาะคอลัมน์ตัวเลข
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        print("num     ", numeric_cols)
        # 2. สร้าง List ของ Min ดิบ และ Max ดิบ
        Ls_raw = [float(df[col].min()) for col in numeric_cols]
        Hs_raw = [float(df[col].max()) for col in numeric_cols]
        
        # 3. ส่งเข้าฟังก์ชันขยายร่าง (Expand) ตามที่คุณนนท์ออกแบบไว้
        Ls_expanded, Hs_expanded = lib.max_min_range_expand(Ls_raw, Hs_raw)
        
        # 4. นำผลลัพธ์ที่ขยายแล้ว มาจัดรูปแบบเข้า JSON แบบแยกคอลัมน์
        min_max_data = {}
        for i, col in enumerate(numeric_cols):
            min_max_data[col] = {
                "min": Ls_expanded[i],
                "max": Hs_expanded[i]
            }
                
        min_max_out = dataset_out_dir / "min_max_scaler.json"
        with open(min_max_out, "w", encoding="utf-8") as f:
            json.dump(min_max_data, f, indent=4)
            
        print(f"✅ [min-max] ขยายร่าง (Expand) และสร้างไฟล์กุญแจสำเร็จ! ตรวจสอบได้ที่: {min_max_out}\n")
    except Exception as e:
        print(f"❌ [min-max] เกิดข้อผิดพลาดในการสร้างไฟล์กุญแจ: {e}\n")

    MODES = [("nopca", False), ("pca", True)]
    # MODES = [("nopca", False)]

    for mode_name, is_pca_mode in MODES:
        print("\n====================================================")
        print(f" RUN MODE: {mode_name}  (is_pca_mode={is_pca_mode})")
        print("====================================================\n")

        lib.remove_and_create_folder(folders)

        if is_pca_mode:
            _path_csv = _pca_csv
            _header_name = header_name_pca
        else:

            _path_csv = _path_csv_raw
            _header_name = _header_name_raw

        if os.path.exists(_output_kfold_data):
            print("NOTE: K-fold file exists. Load:", _output_kfold_data)
            with open(_output_kfold_data, 'r') as f:
                [result_k_fold, train_index_k_fold, validate_index_k_fold, test_feature_k_fold,
                 min_features_k_fold, max_features_k_fold, indices_test, indices_train_validate] = json.load(f)
        else:
            print("CREATE new CROSS VALIDATION")
            (result_k_fold, train_index_k_fold, validate_index_k_fold, test_feature_k_fold,
             min_features_k_fold, max_features_k_fold,
             indices_test, indices_train_validate) = lib.preprocess_create_k_fold(
                _path_csv, _header_name, _label_header
            )
            data_serializable = lib.convert_np_types([
                result_k_fold, train_index_k_fold, validate_index_k_fold, test_feature_k_fold,
                min_features_k_fold, max_features_k_fold, indices_test, indices_train_validate
            ])
            with open(_output_kfold_data, 'w') as f:
                json.dump(data_serializable, f)

        (result_k_fold, train_index_k_fold, validate_index_k_fold, test_feature_k_fold,
         min_features_k_fold, max_features_k_fold,
         indices_test, indices_train_validate) = lib.preprocess_create_k_fold(
            _path_csv, _header_name, _label_header,
            indices_test, indices_train_validate
        )

        for fold_num in range(len(train_index_k_fold)):
            Ls = min_features_k_fold[fold_num]
            Hs = max_features_k_fold[fold_num]
            Ls, Hs = lib.max_min_range_expand(Ls, Hs)

            print("path CSV   ",_path_csv)

            lib.strips_encode(
                _path_csv, _header_name, _label_header, _output_label_list,
                Ls, Hs,
                os.path.join(_output_strip_encode_v1, str(fold_num)),
                os.path.join(_output_strip_encode_label_v1, str(fold_num)),
            )

            print("path strip   ",_output_strip_encode_v1)
            # lib.strips_encode2(
            #     _path_csv, _header_name, _label_header, _output_label_list,
            #     Ls, Hs,
            #     os.path.join(_output_strip_encode_v2, str(fold_num)),
            #     os.path.join(_output_strip_encode_label_v2, str(fold_num)),
            # )
            # lib.strips_encode3(
            #     _path_csv, _header_name, _label_header, _output_label_list,
            #     Ls, Hs,
            #     os.path.join(_output_strip_encode_v3, str(fold_num)),
            #     os.path.join(_output_strip_encode_label_v3, str(fold_num)),
            # )

        lib.preprocess_training_strategy(_path_csv, _output_directory, _output_strip_encode_v1, _output_strip_encode_label_v1, _output_label_list, _output_k_fold_v1, result_k_fold)
        # lib.preprocess_training_strategy(_path_csv, _output_directory, _output_strip_encode_v2, _output_strip_encode_label_v2, _output_label_list, _output_k_fold_v2, result_k_fold)
        # lib.preprocess_training_strategy(_path_csv, _output_directory, _output_strip_encode_v3, _output_strip_encode_label_v3, _output_label_list, _output_k_fold_v3, result_k_fold)

        # LOOP OVER K VALUES FOR BOOST PACKING
        boost_script = os.path.join(_output_directory, "boost_nopca_pack_multi.py")
        
        for k in k_values:
            k_folder = dataset_out_dir / f"repacking_k_{k}"
            ensure_dir(k_folder)

            cmd = [
                sys.executable, boost_script, 
                "--cell", args.cell,
                "--k", str(k),
                "--dest_root", str(k_folder),
                "--dataset_name", dataset_name
            ]
            if is_pca_mode:
                cmd.append("--pca-mode")

            print(f"\n[K={k}] Running:", " ".join(cmd))
            subprocess.run(cmd, check=True)

    print("\n✅ strip_pre2 finished:", dataset_name)