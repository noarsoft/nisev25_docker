# retrain_from_best.py
# Retrain per-fold from existing best.pt using each fold's dataset.yaml
# Example:
# python retrain_from_best.py --base_dirs "C:\yolo8_strip\ultralytics\backup_wine\output_k_fold_boost_nopca" "C:\yolo8_strip\ultralytics\backup_wine\output_k_fold_boost_nopca2" "C:\yolo8_strip\ultralytics\backup_wine\output_k_fold_boost_nopca3" "C:\yolo8_strip\ultralytics\backup_wine\output_k_fold_boost_pca" "C:\yolo8_strip\ultralytics\backup_wine\output_k_fold_boost_pca2" "C:\yolo8_strip\ultralytics\backup_wine\output_k_fold_boost_pca3" --folds 0 1 2 3 4 --device 0 --imgsz 640 --epochs 1000 --batch 2 --workers 1 --name train1 --exist_ok --cos_lr --amp --patience 50


import argparse
import subprocess
import sys
from pathlib import Path

def run(cmd, cwd=None):
    print("\n>>> " + " ".join(map(str, cmd)))
    r = subprocess.run(cmd, cwd=cwd, text=True)
    if r.returncode != 0:
        sys.exit(r.returncode)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dirs", nargs="+", required=True, help="list of base dirs")
    ap.add_argument("--folds", nargs="+", default=["0","1","2","3","4"])
    ap.add_argument("--device", default="0")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--yolo", default="yolo", help="YOLO CLI command, e.g. yolo or python -m ultralytics")
    ap.add_argument("--name", default="retrainA", help="run name under runs/foldX/")
    ap.add_argument("--lr0", type=float, default=None, help="optional override lr0")
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--cos_lr", action="store_true", help="enable cos_lr")
    ap.add_argument("--amp", action="store_true", help="enable amp")
    ap.add_argument("--exist_ok", action="store_true", help="allow overwrite run folder")

    args = ap.parse_args()

    for base_dir in args.base_dirs:
        base = Path(base_dir).resolve()
        if not base.exists():
            print(f"[skip] base_dir not found: {base}")
            continue

        runs_root = base / "runs"
        print(f"\n==============================")
        print(f"BASE: {base}")
        print(f"==============================")

        for fold in args.folds:
            fold = str(fold)
            ds_yaml = base / fold / "dataset.yaml"
            w_best  = runs_root / f"fold{fold}" / "train1" / "weights" / "best.pt"

            if not ds_yaml.exists():
                print(f"[fold {fold}] skip: dataset.yaml not found -> {ds_yaml}")
                continue
            if not w_best.exists():
                print(f"[fold {fold}] skip: best.pt not found -> {w_best}")
                continue

            project_dir = runs_root / f"fold{fold}"
            project_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                args.yolo, "train",
                f"model={str(w_best)}",
                f"data={str(ds_yaml)}",
                f"imgsz={args.imgsz}",
                f"epochs={args.epochs}",
                f"batch={args.batch}",
                f"workers={args.workers}",
                f"device={args.device}",
                f"seed={args.seed}",
                f"project={str(project_dir)}",
                f"name={args.name}",
            ]

            if args.exist_ok:
                cmd.append("exist_ok=True")
            if args.cos_lr:
                cmd.append("cos_lr=True")
            if args.amp:
                cmd.append("amp=True")
            if args.lr0 is not None:
                cmd.append(f"lr0={args.lr0}")
            if args.patience is not None:
                cmd.append(f"patience={args.patience}")

            print(f"[fold {fold}] data={ds_yaml}")
            print(f"[fold {fold}] start_from={w_best}")
            run(cmd)

    print("\n✅ DONE all retrains")

if __name__ == "__main__":
    main()
