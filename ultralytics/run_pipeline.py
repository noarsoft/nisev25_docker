# run_pipeline.py


# python run_pipeline.py --mode both --base_dir "C:\yolo8_strip\ultralytics\output_k_fold_boost_pca" --folds 0 1 2 3 4 --model yolov8s.pt --imgsz 640 --epochs1 1000 --epochs2 1000 --batch 2 --workers 1 --device 0 --datasets adult --oversample_k 1 --validate_after_hard

# -----------------------------------------------------------------------------
# train ปกติ  train.py / plain            -> backup_*/*/train/weights
# train1      pipeline รอบแรก             -> <base_dir>/runs/foldX/train1/weights
# train_hard  pipeline หลัง hard          -> <base_dir>/runs/foldX/train_hard/weights
#
# python run_pipeline.py --mode both --base_dir "C:\yolo8_strip\ultralytics\output_k_fold_boost_pca" --folds 0 1 2 3 4 --model yolov8s.pt --imgsz 640 --epochs1 1000 --epochs2 1000 --batch 2 --workers 1 --device 0 --datasets adult --oversample_k 1 --validate_after_hard

# Key Fix:
#   - If dataset is too small per fold, auto-skip split2 and hard round.
#   - Use original fold dataset.yaml for training (no split, no hard).
#   - Pass --datasets and --project_name to train.py in plain/both mode.
# -----------------------------------------------------------------------------

import argparse, subprocess, sys, shutil
from pathlib import Path

# ------------------------------ utils ------------------------------

def run(cmd, cwd=None):
    print("\n>>>", " ".join(map(str, cmd)))
    r = subprocess.run(cmd, cwd=cwd, text=True)
    if r.returncode != 0:
        sys.exit(r.returncode)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def count_images_in_dir(img_dir: Path) -> int:
    """
    Count image files recursively in a directory.
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    if not img_dir.exists():
        return 0
    return sum(1 for p in img_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts)

def read_txt_nonempty_lines(p: Path) -> int:
    """
    Return number of non-empty lines in a txt list file (if exists).
    """
    if not p.exists():
        return 0
    try:
        lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
        return sum(1 for x in lines if x.strip())
    except Exception:
        return 0

# ------------------------------ pipeline per fold ------------------------------

def do_pipeline_for_fold(args, base: Path, fold: str, scripts_dir: Path):
    fold_dir = base / str(fold)

    # original dataset (works when it has training/validation folders)
    ds_yaml_orig = fold_dir / "dataset.yaml"

    # split2 dataset
    splits_dir = fold_dir / "_splits2"
    ds_yaml2   = splits_dir / "dataset2.yaml"
    ds_hard    = splits_dir / "dataset_with_hard.yaml"

    # decide "small data" based on number of training images
    n_train_imgs = count_images_in_dir(fold_dir / "training" / "images")
    small_data = (n_train_imgs < args.min_images_for_split)

    # fallback mode: small dataset -> no split, no hard
    if small_data:
        print(f"[fold {fold}] small data detected: training images={n_train_imgs} < min_images_for_split={args.min_images_for_split}")
        print(f"[fold {fold}] -> use original dataset.yaml and skip split2 + hard round.")
        use_yaml = ds_yaml_orig
        do_hard = False
    else:
        use_yaml = ds_yaml2
        do_hard = (not args.skip_hard_round)

        # 1) split (only when data is enough)
        if not args.skip_split:
            split_py = scripts_dir / "split_dataset.py"
            run([
                sys.executable, str(split_py),
                "--base_dir", str(base),
                "--folds", *map(str, args.folds)
            ], cwd=scripts_dir)

        # safety check: ensure val2 has at least 1 image in the list file
        val2_txt = splits_dir / "val2.txt"
        n_val2 = read_txt_nonempty_lines(val2_txt)
        if n_val2 == 0:
            print(f"[fold {fold}] WARNING: val2.txt is empty after split. Fallback to original dataset.yaml, skip hard.")
            use_yaml = ds_yaml_orig
            do_hard = False

    # project dir per fold (pipeline outputs)
    project_dir = base / "runs" / f"fold{fold}"
    ensure_dir(project_dir)

    # batch fallback
    batch1 = args.batch if args.batch != -1 else 2

    # 2) train round 1
    run([
        args.yolo, "train",
        f"model={args.model}",
        f"data={str(use_yaml)}",
        f"imgsz={args.imgsz}",
        f"epochs={args.epochs1}",
        f"batch={batch1}",
        f"workers={args.workers}",
        "cos_lr=True", "amp=True",
        f"device={args.device}", f"seed={args.seed}",
        f"project={str(project_dir)}", "name=train1", "exist_ok=True",
    ], cwd=scripts_dir)

    # if no hard (small dataset) -> finish
    if not do_hard:
        print(f"\n==== Fold {fold} finished (no split/hard) ====\n")
        return

    # 3) val on test2 to generate predictions for mining hard
    best1 = project_dir / "train1" / "weights" / "best.pt"
    run([
        args.yolo, "val",
        f"model={str(best1)}",
        f"data={str(ds_yaml2)}",
        "split=test", "save_txt", "save_conf",
        f"project={str(project_dir)}", "name=val_test2_round1", "exist_ok=True",
    ], cwd=scripts_dir)

    # 4) mine hard
    mine_py = scripts_dir / "mine_hard_from_test.py"
    run([
        sys.executable, str(mine_py),
        "--test2_list", str(splits_dir / "test2.txt"),
        "--gt_labels_root", str(fold_dir / "training" / "labels"),
        "--runs_detect_dir", str(project_dir / "val_test2_round1"),
        "--out_hard_txt", str(splits_dir / "hard_from_test2.txt"),
        "--conf_threshold", str(args.conf_threshold),
    ], cwd=scripts_dir)

    # 5) make dataset_with_hard
    make_py = scripts_dir / "make_train_with_hard.py"
    run([
        sys.executable, str(make_py),
        "--splits_dir", str(splits_dir),
        "--oversample_k", str(args.oversample_k),
        "--seed", str(args.seed),
    ], cwd=scripts_dir)

    # 6) train hard round
    run([
        args.yolo, "train",
        f"model={str(best1)}",
        f"data={str(ds_hard)}",
        f"imgsz={args.imgsz}",
        f"epochs={args.epochs2}",
        "batch=1",  # focus hard samples
        "cos_lr=True", "amp=True",
        f"workers={args.workers}",
        f"device={args.device}", f"seed={args.seed}",
        f"project={str(project_dir)}", "name=train_hard", "exist_ok=True",
    ], cwd=scripts_dir)

    # 7) optional validate after hard
    if args.validate_after_hard:
        best2 = project_dir / "train_hard" / "weights" / "best.pt"
        run([
            args.yolo, "val",
            f"model={str(best2)}",
            f"data={str(ds_hard)}",
            "split=test", "save_txt", "save_conf",
            f"project={str(project_dir)}", "name=val_test2_round2", "exist_ok=True",
        ], cwd=scripts_dir)

    print(f"\n==== Fold {fold} finished (pipeline) ====\n")

# ------------------------------ main ------------------------------

def main():
    ap = argparse.ArgumentParser(description="YOLO hard-example pipeline across folds + plain (train.py) option")

    ap.add_argument("--base_dir", required=True, help=r"ex. C:\yolo8_strip\ultralytics\output_k_fold2_boost")
    ap.add_argument("--folds", nargs="+", default=["0","1","2","3","4"])

    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs1", type=int, default=1000, help="epochs สำหรับ train รอบแรก หรือ plain")
    ap.add_argument("--epochs2", type=int, default=1000, help="epochs สำหรับ train รอบ hard")

    ap.add_argument("--batch", type=int, default=-1, help="batch size สำหรับ train (plain/pipeline รอบแรก) (-1 จะ fallback เป็น 2)")
    ap.add_argument("--workers", type=int, default=0)

    ap.add_argument("--oversample_k", type=int, default=1)
    ap.add_argument("--device", default="0")
    ap.add_argument("--yolo", default="yolo", help="YOLO CLI (ex 'yolo' or 'python -m ultralytics')")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--conf_threshold", type=float, default=0.5)

    ap.add_argument("--validate_after_hard", action="store_true")
    ap.add_argument("--skip_split", action="store_true")
    ap.add_argument("--skip_hard_round", action="store_true")

    ap.add_argument("--clean-first", action="store_true",
                    help="delete _splits2 and runs/foldX before running")

    ap.add_argument("--mode", choices=["pipeline", "plain", "both"], default="pipeline",
                    help="เลือก flow การฝึก: pipeline (เดิม), plain (ไม่ split ผ่าน train.py), หรือ both")

    # for plain/both -> passed to train.py
    ap.add_argument("--datasets", type=str, default="adult",
                    help="dataset name(s) ส่งต่อให้ train.py เช่น adult,iris (คั่นด้วยคอมมาได้)")
    ap.add_argument("--plain_project_name", type=str, default=None,
                    help="ตั้งชื่อ backup_* โดยตรง (ส่งต่อไป train.py --project_name)")

    # threshold for deciding to do split2/hard
    ap.add_argument("--min_images_for_split", type=int, default=5,
                    help="ถ้าจำนวน training images ใน fold < ค่านี้ จะไม่ทำ split2/hard และจะใช้ dataset.yaml เดิม")

    args = ap.parse_args()

    base = Path(args.base_dir).resolve()
    scripts_dir = Path(__file__).parent.resolve()

    # clean-first: remove per-fold splits + runs (safe)
    if args.clean_first:
        runs_root = base / "runs"
        for f in args.folds:
            foldsplits = base / str(f) / "_splits2"
            foldruns   = runs_root / f"fold{f}"
            for p in (foldsplits, foldruns):
                if p.exists():
                    print(f"[clean] removing {p}")
                    shutil.rmtree(p, ignore_errors=True)
        print("✅ Clean-first done.\n")

    # ----- plain mode: call train.py once for all folds -----
    if args.mode in ("plain", "both"):
        train_py = scripts_dir / "train.py"
        if not train_py.exists():
            print(f"[plain] ERROR: not found train.py at {train_py}")
            sys.exit(1)

        folds_csv = ",".join(map(str, args.folds))
        cmd = [
            sys.executable, str(train_py),
            "--weights", str(args.model),
            "--epochs", str(args.epochs1),
            "--img", str(args.imgsz),
            "--batch", str(args.batch if args.batch != -1 else 2),
            "--workers", str(args.workers),
            "--device", str(args.device),
            "--folds", folds_csv,
            "--base_dir", str(base),
            "--datasets", str(args.datasets),   # ✅ ส่ง dataset ไป train.py
        ]
        if args.plain_project_name:
            cmd += ["--project_name", str(args.plain_project_name)]  # ✅ override ชื่อ backup_*

        print("\n[plain] calling train.py ...")
        run(cmd, cwd=scripts_dir)
        print("[plain] finished.\n")

    # ----- pipeline mode: per fold pipeline (split2 + hard) with fallback -----
    if args.mode in ("pipeline", "both"):
        for fold in args.folds:
            do_pipeline_for_fold(args, base, str(fold), scripts_dir)

if __name__ == "__main__":
    main()
