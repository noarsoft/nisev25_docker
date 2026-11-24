#strip_pre2_back.py

import lib
import os
import sys
import json
import subprocess


# prequisition
# u need to setup yolov8 and can runable, firstly


# --THIS FILE is preprocessing and creating strip encode from k-fold--
if __name__ == "__main__":

    _output_directory = os.path.dirname(os.path.abspath(__file__))
    _output_label_list = os.path.join(_output_directory, "output_strip_label_map.txt")
    _output_kfold_data = os.path.join(_output_directory, "output_kfold_data.json")

    # algor1
    _output_strip_encode_v1 = os.path.join(_output_directory, "out_strip")
    _output_strip_encode_label_v1 = os.path.join(_output_directory, "out_strip_label")
    _output_k_fold_v1 = os.path.join(_output_directory, "output_k_fold")
    _output_k_fold_v1_boost = os.path.join(_output_k_fold_v1 + "_boost")

    # algor2
    _output_strip_encode_v2 = os.path.join(_output_directory, "out_strip2")
    _output_strip_encode_label_v2 = os.path.join(_output_directory, "out_strip_label2")
    _output_k_fold_v2 = os.path.join(_output_directory, "output_k_fold2")
    _output_k_fold_v2_boost = os.path.join(_output_k_fold_v2 + "_boost")

    # algor3
    _output_strip_encode_v3 = os.path.join(_output_directory, "out_strip3")
    _output_strip_encode_label_v3 = os.path.join(_output_directory, "out_strip_label3")
    _output_k_fold_v3 = os.path.join(_output_directory, "output_k_fold3")
    _output_k_fold_v3_boost = os.path.join(_output_k_fold_v3 + "_boost")

    folders = [
        _output_strip_encode_v1, _output_strip_encode_label_v1, _output_k_fold_v1, _output_k_fold_v1_boost,
        _output_strip_encode_v2, _output_strip_encode_label_v2, _output_k_fold_v2, _output_k_fold_v2_boost,
        _output_strip_encode_v3, _output_strip_encode_label_v3, _output_k_fold_v3, _output_k_fold_v3_boost,
    ]

    # ======== step:0 select dataset ========

    # https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset
    # _path_csv_raw, _header_name_raw, _label_header, _pca_csv, header_name_pca = lib.load_header_student_performance(_output_directory)

    # _path_csv_raw, _header_name_raw, _label_header, _pca_csv, header_name_pca = lib.load_header_breast_cancer(_output_directory)

    # ****--load iris-- https://archive.ics.uci.edu/
    # _path_csv_raw, _header_name_raw, _label_header, _pca_csv, header_name_pca = lib.load_header_iris(_output_directory)

    # ****--load wine 3 class--
    _path_csv_raw, _header_name_raw, _label_header, _pca_csv, header_name_pca = lib.load_wine(_output_directory)

    # https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset
    # _path_csv_raw, _header_name_raw, _label_header, _pca_csv, header_name_pca = lib.load_header_student_performance(_output_directory)

    # load_header_breast_cancer
    # _path_csv_raw, _header_name_raw, _label_header, _pca_csv, header_name_pca = lib.load_header_breast_cancer(_output_directory)

    # https://www.kaggle.com/datasets/uciml/adult-census-income
    # _path_csv_raw, _header_name_raw, _label_header, _pca_csv, header_name_pca = lib.load_header_adult(_output_directory)

    # https://www.kaggle.com/datasets/ulrikthygepedersen/electricity-demands
    # _path_csv_raw, _header_name_raw, _label_header, _pca_csv, header_name_pca = lib.load_header_electric(_output_directory)

    # ======== main loop: run both nopca and pca ========
    MODES = [
        ("nopca", False),
        ("pca",   True),
    ]

    for mode_name, is_pca_mode in MODES:
        print("\n====================================================")
        print(f" RUN MODE: {mode_name}  (is_pca_mode={is_pca_mode})")
        print("====================================================\n")

        # ล้างโฟลเดอร์ output ชั่วคราว เหมือนเริ่มรันใหม่ทุกโหมด
        lib.remove_and_create_folder(folders)

        # ======== step:0.1 OPTIONAL PCA ========
        if is_pca_mode:
            _path_csv = _pca_csv
            _header_name = header_name_pca
        else:
            _path_csv = _path_csv_raw
            _header_name = _header_name_raw

        # ======== step 1: create k-fold and explore min and max from features =========
        print("======== step 1: create k-fold and explore min and max from features =========")

        if os.path.exists(_output_kfold_data):
            # load crossvalidation (reuse split เดิม)
            print(
                "NOTE: K-fold file existing " + _output_kfold_data + ". Load K-fold file. "
                "if want to renew data, delete the file."
            )
            print(
                "This might error, if data in " + _output_kfold_data +
                " is not macth with the image in strip folder this need be renewed data. (delete the file)"
            )
            with open(_output_kfold_data, 'r') as f:
                [result_k_fold, train_index_k_fold, validate_index_k_fold, test_feature_k_fold,
                 min_features_k_fold, max_features_k_fold, indices_test, indices_train_validate] = json.load(f)
        else:
            # create cross validation (ครั้งแรกเท่านั้น)
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

        # run preprocess_create_k_fold อีกทีด้วย indices เดิม เพื่อให้ได้ min/max ตามโหมดนี้
        (result_k_fold, train_index_k_fold, validate_index_k_fold, test_feature_k_fold,
         min_features_k_fold, max_features_k_fold,
         indices_test, indices_train_validate) = lib.preprocess_create_k_fold(
            _path_csv, _header_name, _label_header,
            indices_test, indices_train_validate
        )

        # ======== step 2: create strip encoder from the K-fold ========
        print("======== step 2: create strip encoder from the K-fold ========")
        for fold_num in range(len(train_index_k_fold)):
            output_strip = os.path.join(_output_strip_encode_v1, str(fold_num))
            output_strip_label = os.path.join(_output_strip_encode_label_v1, str(fold_num))

            output_strip2 = os.path.join(_output_strip_encode_v2, str(fold_num))
            output_strip_label2 = os.path.join(_output_strip_encode_label_v2, str(fold_num))

            output_strip3 = os.path.join(_output_strip_encode_v3, str(fold_num))
            output_strip_label3 = os.path.join(_output_strip_encode_label_v3, str(fold_num))

            Ls = min_features_k_fold[fold_num]
            Hs = max_features_k_fold[fold_num]
            Ls, Hs = lib.max_min_range_expand(Ls, Hs)

            print('---create strip encoder algor1 (K-fold=' + str(fold_num) + ')---')
            lib.strips_encode(_path_csv, _header_name, _label_header, _output_label_list,
                              Ls, Hs, output_strip, output_strip_label)

            print('---create strip encoder algor2 (K-fold=' + str(fold_num) + ')---')
            lib.strips_encode2(_path_csv, _header_name, _label_header, _output_label_list,
                               Ls, Hs, output_strip2, output_strip_label2)

            print('---create strip encoder algor3 (K-fold=' + str(fold_num) + ')---')
            lib.strips_encode3(_path_csv, _header_name, _label_header, _output_label_list,
                               Ls, Hs, output_strip3, output_strip_label3)

        # ======== step 3: preprocess_training_strategy ======== (cross validation)
        print("======== step 3: preprocess_training_strategy ======== (cross validation)")

        # algor1
        yamls, test_folders_img, test_folders_label, result_k_fold = \
            lib.preprocess_training_strategy(_path_csv, _output_directory, _output_strip_encode_v1,
                                             _output_strip_encode_label_v1, _output_label_list,
                                             _output_k_fold_v1, result_k_fold)

        # algor2
        yamls, test_folders_img, test_folders_label, result_k_fold = \
            lib.preprocess_training_strategy(_path_csv, _output_directory, _output_strip_encode_v2,
                                             _output_strip_encode_label_v2, _output_label_list,
                                             _output_k_fold_v2, result_k_fold)

        # algor3
        yamls, test_folders_img, test_folders_label, result_k_fold = \
            lib.preprocess_training_strategy(_path_csv, _output_directory, _output_strip_encode_v3,
                                             _output_strip_encode_label_v3, _output_label_list,
                                             _output_k_fold_v3, result_k_fold)

        print("--finished preprocessing step for mode:", mode_name)

        # ======== step 4: create 640x640 boost images by calling boost_nopca_pack_multi.py ========
        print("======== step 4: create packed boost images (640x640) for mode", mode_name, "========")

        boost_script = os.path.join(_output_directory, "boost_nopca_pack_multi.py")

        cmd = [sys.executable, boost_script, "--cell", "auto"]
        if is_pca_mode:
            cmd.append("--pca-mode")

        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        print("Finished boost packing for mode:", mode_name)

    print(
        "======== ALL MODES FINISHED ========\n"
        "Next step: train YOLO models from output_k_fold_boost_nopca* and output_k_fold_boost_pca* folders.\n"
        "See each fold's train.bat or dataset.yaml for training commands.\n"
    )
