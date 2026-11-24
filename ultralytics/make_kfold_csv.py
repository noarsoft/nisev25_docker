# make_kfold_csv.py
#python make_kfold_csv.py --dataset adult

import os
import argparse
import lib

DATASET_LOADERS = {
    "adult": lib.load_k_float_encoder_adult,
    "iris": lib.load_k_float_encoder_iris,
    "wine": lib.load_k_float_encoder_wine,
    "yeast": lib.load_k_float_encoder_yeast,
    "nutrition": lib.load_k_float_encoder_nutrition,
    "electric": lib.load_k_float_encoder_electric,
    "student": lib.load_k_float_encoder_student_performance,
    "ionosphere": lib.load_k_float_encoder_ionosphere,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=DATASET_LOADERS.keys(),
                    help="ชื่อ dataset เช่น adult, iris, wine, yeast, nutrition, electric, student")
    args = ap.parse_args()

    project_dir = os.path.dirname(os.path.abspath(__file__))

    loader = DATASET_LOADERS[args.dataset]
    result, label_name = loader(project_dir)

    print(f"[{args.dataset}] label column name:", label_name)
    print("train/val/test csv created at these paths:")
    for train_csv, val_csv, test_csv in result:
        print("  train :", train_csv)
        print("  val   :", val_csv)
        print("  test  :", test_csv)
        print("-----")
