# train.py
# python train.py --weights yolov8s.pt --epochs 1000 --img 640 --batch 2 --workers 1 --device 0 --algos 0,1,2 --folds 0,1,2,3,4
# หรือเทรนปกติแบบไม่ split จาก base เดียว:
# python train.py --weights yolov8s.pt --epochs 300 --img 640 --batch 2 --workers 1 --device 0 --folds 0,1,2,3,4 --base_dir "C:\yolo8_strip\ultralytics\output_k_fold2_boost"

import argparse
import csv
import os
import time
from pathlib import Path

import torch
from ultralytics.models.yolo.detect import DetectionTrainer

print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    try:
        print("GPU[0]:", torch.cuda.get_device_name(0))
    except Exception:
        pass

ALGOS = [
    { "boost_dir": "output_k_fold_boost_nopca", "result_prefix": "backup_{data}_boost_nopca_algor1" },
    { "boost_dir": "output_k_fold_boost_nopca2", "result_prefix": "backup_{data}_boost_nopca_algor2" },
    { "boost_dir": "output_k_fold_boost_nopca3", "result_prefix": "backup_{data}_boost_nopca_algor3" },
]

DATASETS = ["adult"]
DEFAULT_FOLDS = ["0", "1", "2", "3", "4"]

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def train_one_fold(yaml_path: str, out_dir: str, opt):
    args = {
        "model": opt.weights,
        "data": yaml_path,
        "epochs": opt.epochs,
        "imgsz": opt.img,
        "batch": opt.batch,
        "workers": opt.workers,
        "device": opt.device,
        "project": out_dir,
    }
    trainer = DetectionTrainer(overrides=args)
    t0 = time.time()
    trainer.train()
    t1 = time.time()
    return t1 - t0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--img", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--folds", type=str, default=",".join(DEFAULT_FOLDS))
    parser.add_argument("--algos", type=str, default="0,1,2")
    parser.add_argument("--datasets", type=str, default=",".join(DATASETS))
    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument("--project_name", type=str, default=None)
    opt = parser.parse_args()

    folds = [f.strip() for f in opt.folds.split(",") if f.strip() != ""]
    datasets = [d.strip() for d in opt.datasets.split(",") if d.strip() != ""]

    root_dir = os.path.dirname(os.path.abspath(__file__))

    if opt.base_dir is not None:
        base_dir = Path(opt.base_dir).resolve()
        base_name = base_dir.name
        
        for data_name in datasets:
            result_root = (Path(root_dir) / (opt.project_name if opt.project_name else f"backup_{data_name}_{base_name}"))
            ensure_dir(result_root)

            print(f"\n=== DATASET: {data_name} | BASE_DIR: {base_dir} → results at {result_root} ===")

            csv_path = result_root / "train_times.csv"
            if not csv_path.exists():
                ensure_dir(csv_path.parent)
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Fold", "Start Time", "End Time", "Elapsed Seconds", "Elapsed HH:MM:SS", "Algo", "Dataset"])

            for fold in folds:
                yaml_path = base_dir / str(fold) / "dataset.yaml"
                if not yaml_path.exists():
                    print(f"!! skip fold {fold}: not found {yaml_path}")
                    continue

                out_dir = result_root / fold
                ensure_dir(out_dir)
                print(f"\n[FOLD {fold}] training from {yaml_path}")
                start_time = time.time()
                elapsed = train_one_fold(str(yaml_path), str(out_dir), opt)
                end_time = time.time()

                h = int(elapsed // 3600)
                m = int((elapsed % 3600) // 60)
                s = int(elapsed % 60)
                elapsed_hms = f"{h:02d}:{m:02d}:{s:02d}"

                print(f"[FOLD {fold}] finished. Time used: {elapsed_hms}")

                with open(csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        fold, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
                        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)),
                        f"{elapsed:.2f}", elapsed_hms, "base_dir", data_name
                    ])

        print("\n✅ All training loops finished (base_dir mode).")
        return

if __name__ == "__main__":
    main()