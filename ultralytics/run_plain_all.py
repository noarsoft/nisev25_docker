# run_plain_all.py
import subprocess
import sys
import argparse
import shutil
import time
from pathlib import Path
import json
import yaml

DEFAULT_BASE_DIRS_REL = [
    "output_k_fold_boost_nopca",
    "output_k_fold_boost_nopca2",
    "output_k_fold_boost_nopca3",
    "output_k_fold_boost_pca",
    "output_k_fold_boost_pca2",
    "output_k_fold_boost_pca3",
]

DEFAULT_FOLDS = ["0", "1", "2", "3", "4"]

SCRIPT_DIR = Path(__file__).parent.resolve()
TRAIN_PY = SCRIPT_DIR / "train.py"

def run(cmd):
    print("\n>>>", " ".join(map(str, cmd)))
    r = subprocess.run(cmd, cwd=SCRIPT_DIR)
    if r.returncode != 0:
        sys.exit(r.returncode)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def rm_tree(p: Path):
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--dataset_root", default=r"C:\yolo8_strip\ultralytics")
    ap.add_argument("--base_dirs_rel", nargs="+", default=DEFAULT_BASE_DIRS_REL)
    ap.add_argument("--folds", nargs="+", default=DEFAULT_FOLDS)
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project_prefix", default="plain")
    ap.add_argument("--clean_output", action="store_true")
    ap.add_argument("--clean_each_project", action="store_true")
    ap.add_argument("--exist_ok", action="store_true")
    ap.add_argument("--k_list", default="0", help="comma-separated K values e.g. 0,2,4")

    args = ap.parse_args()

    if not TRAIN_PY.exists():
        print(f"train.py not found at {TRAIN_PY}")
        sys.exit(1)

    dataset_root = Path(args.dataset_root).resolve()
    ds_out_root = dataset_root / args.datasets
    ensure_dir(ds_out_root)
    
    k_values = [x.strip() for x in args.k_list.split(",") if x.strip() != ""]

    if args.clean_output:
        for k in k_values:
            for rel in args.base_dirs_rel:
                name = Path(rel).name
                project_dir = ds_out_root / f"repacking_k_{k}" / f"{args.project_prefix}_{name}"
                rm_tree(project_dir)

    summary_path = ds_out_root / "plain_training_summary.csv"
    summary_lines = ["dataset,k_value,project,base_dir_rel,seconds_total,avg_seconds_per_epoch,epochs,batch,imgsz,device,folds\n"]

    for k in k_values:
        for rel in args.base_dirs_rel:
            name = Path(rel).name
            base_dir = ds_out_root / f"repacking_k_{k}" / rel
            
            if not base_dir.exists():
                print(f"\n[Skip] base_dir not found for K={k}: {base_dir}")
                continue

            project_name = f"{args.project_prefix}_{name}"
            project_dir = ds_out_root / f"repacking_k_{k}" / project_name

            if args.clean_each_project:
                rm_tree(project_dir)

            print("\n========================================")
            print(f"PLAIN TRAINING [K = {k}]")
            print(f"  dataset : {args.datasets}")
            print(f"  base_dir: {base_dir}")
            print(f"  output  : {project_dir}")
            print("========================================")

            ensure_dir(project_dir)
            
            # บังคับให้เซฟไฟล์เข้าโฟลเดอร์ย่อย dataset/repacking_k_../
            nested_project_name = str(Path(args.datasets) / f"repacking_k_{k}" / project_name).replace("/", "\\")

            cmd = [
                sys.executable, str(TRAIN_PY),
                "--weights", args.model,
                "--epochs", str(args.epochs),
                "--img", str(args.imgsz),
                "--batch", str(args.batch),
                "--workers", str(args.workers),
                "--device", str(args.device),
                "--folds", ",".join(args.folds),
                "--base_dir", str(base_dir),
                "--datasets", args.datasets,
                "--project_name", nested_project_name,
            ]

            t0 = time.time()
            run(cmd)
            t1 = time.time()

            # =========================================================================
            # 🌟 โค้ดส่วนที่เพิ่ม: ค้นหาและคัดลอก Hyperparameters หลังเทรนเสร็จทุก Fold
            # =========================================================================
            print(f"⏳ กำลังบันทึก Hyperparameters สำหรับ {project_name} ...")
            
            # วิ่งหาไฟล์ args.yaml ในทุกๆ โฟลเดอร์ย่อยของการรันรอบนี้ (เช่น 0/train/args.yaml)
            for args_yaml_path in project_dir.rglob("args.yaml"):
                save_dir = args_yaml_path.parent
                weights_dir = save_dir / "weights"

                if weights_dir.exists() and weights_dir.is_dir():
                    try:
                        # 1. ก๊อปปี้ไฟล์ต้นฉบับ yaml ไปใส่ใน weights
                        target_yaml = weights_dir / "saved_hyperparameters.yaml"
                        shutil.copy2(args_yaml_path, target_yaml)

                        # 2. แปลงข้อมูลและบันทึกเป็นไฟล์ .json ควบคู่ไปด้วย
                        target_json = weights_dir / "saved_hyperparameters.json"
                        with open(args_yaml_path, 'r', encoding='utf-8') as f:
                            args_data = yaml.safe_load(f)
                        with open(target_json, 'w', encoding='utf-8') as f:
                            json.dump(args_data, f, indent=4, ensure_ascii=False)
                            
                    except Exception as e:
                        print(f"⚠️ บันทึก Hyperparameters สำหรับ {save_dir.name} ไม่สำเร็จ: {e}")

            print(f"✅ บันทึก Hyperparameters เสร็จสิ้น")
            # =========================================================================

            sec_total = t1 - t0
            avg_epoch = sec_total / max(1, int(args.epochs))

            summary_lines.append(
                f"{args.datasets},{k},{project_name},{name},{sec_total:.3f},{avg_epoch:.6f},"
                f"{args.epochs},{args.batch},{args.imgsz},{args.device},\"{','.join(args.folds)}\"\n"
            )

            summary_path.write_text("".join(summary_lines), encoding="utf-8")

    print("\nALL PLAIN TRAINING FINISHED")
    print("Summary:", summary_path)

if __name__ == "__main__":
    main()