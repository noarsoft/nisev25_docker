# run_all_datasets.py
#
# Example:
#   python run_all_datasets.py --datasets wine,iris,breast_cancer,student,adult --k_list 0,2,4,8,16 --batch 4 --device 0 --clean_each_project --exist_ok
#   python run_all_datasets.py --datasets iris --k_list 0,2,4,8,16 --batch 6 --device 0 --clean_each_project --exist_ok
#   python run_all_datasets.py --datasets iris --k_list 0,2,4 --batch 6 --device 0 --clean_each_project --exist_ok
#   python run_all_datasets.py --datasets iris --k_list 8,16 --epochs 1000 --batch 6 --workers 4 --device 0 --exist_ok
#   python run_all_datasets.py --datasets wine --k_list 0,2,4,8,16 --batch 6 --device 0 --clean_each_project --exist_ok
#   python run_all_datasets.py --datasets breast_cancer --k_list 0,2,4,8,16 --batch 6 --device 0 --clean_each_project --exist_ok
#   python run_all_datasets.py --datasets automobile --k_list 0,2,4,8,16 --batch 6 --device 0 --clean_each_project --exist_ok
#   python run_all_datasets.py --datasets hcc --k_list 0,2,4,8,16 --batch 6 --device 0 --clean_each_project --exist_ok
#   python run_all_datasets.py --datasets ionosphere --k_list 0,2,4,8,16 --batch 6 --device 0 --clean_each_project --exist_ok
#   python run_all_datasets.py --datasets amphibians_green_frogs,amphibians_brown_frogs,amphibians_common_toad,amphibians_fire_bellied_toad,amphibians_tree_frog,amphibians_common_newt,amphibians_great_crested_newt --k_list 8 --batch 6 --device 0 --clean_each_project --exist_ok
#   python run_all_datasets.py --datasets amphibians_brown_frogs --k_list 4 --batch 6 --device 0 --clean_each_project --exist_ok
# Flow per dataset:
#   0) ensure dataset folder exists: ultralytics/<ds>/
#   1) copy backup files:
#       ultralytics/backup_<ds>/output_kfold_data.json  -> ultralytics/output_kfold_data.json
#       ultralytics/backup_<ds>/output_strip_label_map.txt -> ultralytics/output_strip_label_map.txt
#   2) copy dataset csv from ultralytics root -> ultralytics/<ds>/<expected_filename>
#   3) python strip_pre2.py --datasets <ds> --cell <cell>
#   4) python run_plain_all.py --datasets <ds> ... (stores outputs under ultralytics/<ds>/...)

# run_all_datasets.py
import argparse
import subprocess
import sys   
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
STRIP_PRE2 = SCRIPT_DIR / "strip_pre2.py"
RUN_PLAIN  = SCRIPT_DIR / "run_plain_all.py"

EXPECTED_CSV = {
    "wine": "wine.csv",
    "iris": "iris.csv",
    "breast_cancer": "breast-cancer-wisconsin1.csv",
    "student": "Student_performance_data.csv",
    "adult": "adult.csv",
    "electric": "electricity.csv",
    "automobile": "automobile.csv",        
    "hcc": "hcc.csv",
    "ionosphere":"ionosphere.csv",
    
    # amphibians
    "amphibians_green_frogs": "amphibians.csv",
    "amphibians_brown_frogs": "amphibians.csv",
    "amphibians_common_toad": "amphibians.csv",
    "amphibians_fire_bellied_toad": "amphibians.csv",
    "amphibians_tree_frog": "amphibians.csv",
    "amphibians_common_newt": "amphibians.csv",
    "amphibians_great_crested_newt": "amphibians.csv",


    
    # โครงการ 11 ตัว
    "heritage": "export.csv",
    "way_of_life": "export.csv",
    "narrative_symbol": "export.csv",
    "tradition_culture": "export.csv",
    "food_spa_herbs": "export.csv",
    "values_belief_wisdom": "export.csv",
    "crafts_community_products": "export.csv",
    "symbols_indigenous": "export.csv",
    "culture_architecture": "export.csv",
    "traditional_resources": "export.csv",
    "craft_knowledge": "export.csv"
}

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def copy_required(src: Path, dst: Path, label: str):
    if not src.exists():
        print(f"missing required {label}: {src}")
        sys.exit(1)
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    print(f"[backup] copied {label} -> {dst}")

def run(cmd):
    print("\n>>>", " ".join(map(str, cmd)))
    r = subprocess.run(cmd, cwd=SCRIPT_DIR)
    if r.returncode != 0:
        sys.exit(r.returncode)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="wine")
    ap.add_argument("--dataset_root", default=r"C:\yolo8_strip\ultralytics")
    ap.add_argument("--cell", default="auto")
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--device", default="0")
    ap.add_argument("--clean_output", action="store_true")
    ap.add_argument("--clean_each_project", action="store_true")
    ap.add_argument("--exist_ok", action="store_true")
    ap.add_argument("--list_datasets", action="store_true")
    ap.add_argument("--k_list", default="0", help="comma-separated K values e.g. 0,2,4,8,16")

    args = ap.parse_args()

    if args.list_datasets:
        print("Supported datasets:")
        for k, v in EXPECTED_CSV.items():
            print(f"  - {k:14s} expects CSV name: {v}")
        sys.exit(0)

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    if not datasets:
        print("❌ No datasets provided")
        sys.exit(1)

    dataset_root = Path(args.dataset_root).resolve()

    if not STRIP_PRE2.exists():
        sys.exit(f" not found: {STRIP_PRE2}")
    if not RUN_PLAIN.exists():
        sys.exit(f" not found: {RUN_PLAIN}")

    for ds in datasets:
        if ds not in EXPECTED_CSV:
            sys.exit(f" Unknown dataset: {ds} (use --list_datasets)")

        print("\n====================================================")
        print(f"RUN DATASET: {ds}")
        print("====================================================")

        ds_dir = dataset_root / ds
        ensure_dir(ds_dir)

        backup_dir = dataset_root / "backup" / ds
        print(f"\n[backup] dir = {backup_dir}")

        root_kfold = dataset_root / "output_kfold_data.json"
        root_label = dataset_root / "output_strip_label_map.txt"
        bk_kfold = backup_dir / "output_kfold_data.json"
        bk_label = backup_dir / "output_strip_label_map.txt"

        if args.clean_each_project:
            if root_kfold.exists():
                root_kfold.unlink()
            if root_label.exists():
                root_label.unlink()
        else:
            if bk_kfold.exists():
                shutil.copy2(bk_kfold, root_kfold)
            if bk_label.exists():
                shutil.copy2(bk_label, root_label)

        expected = EXPECTED_CSV[ds]
        src_csv = dataset_root / expected
        dst_csv = ds_dir / expected

        if not src_csv.exists():
            hits = list(dataset_root.glob("*.csv"))
            fallback = None
            for p in hits:
                if ds.lower() in p.name.lower():
                    fallback = p
                    break
            if fallback is None:
                sys.exit(f"CSV not found for dataset '{ds}'. Expected: {src_csv}")
            print(f"[csv] expected not found, using fallback: {fallback.name}")
            src_csv = fallback

        shutil.copy2(src_csv, dst_csv)
        print(f"[csv] copied {src_csv.name} -> {dst_csv}")

        src_csv_alt = src_csv.parent / f"_{src_csv.name}"
        dst_csv_alt = dst_csv.parent / f"_{src_csv.name}"
        if src_csv_alt.exists():
            shutil.copy2(src_csv_alt, dst_csv_alt)

        # 2) preprocess with k_list
        run([
            sys.executable, str(STRIP_PRE2),
            "--datasets", ds,
            "--cell", args.cell,
            "--dataset_root", str(dataset_root),
            "--k_list", args.k_list
        ])

        ensure_dir(backup_dir)
        if root_kfold.exists():
            shutil.copy2(root_kfold, bk_kfold)
        if root_label.exists():
            shutil.copy2(root_label, bk_label)

        # 3) plain training
        cmd_plain = [
            sys.executable, str(RUN_PLAIN),
            "--datasets", ds,
            "--dataset_root", str(dataset_root),
            "--model", args.model,
            "--imgsz", str(args.imgsz),
            "--epochs", str(args.epochs),
            "--batch", str(args.batch),
            "--workers", str(args.workers),
            "--device", str(args.device),
            "--k_list", args.k_list
        ]

        if args.clean_output:
            cmd_plain.append("--clean_output")
        if args.clean_each_project:
            cmd_plain.append("--clean_each_project")
        if args.exist_ok:
            cmd_plain.append("--exist_ok")

        run(cmd_plain)

    print("\n ALL DONE (preprocess + plain training)")

if __name__ == "__main__":
    main()