# split_dataset.py
import os
import yaml
import random
import argparse
from pathlib import Path

# config
RATIOS = (0.7, 0.15, 0.15)  # train2 / val2 / test2
SEED = 42
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

random.seed(SEED)

def find_label_for_image(img_path, labels_root_guess) -> Path:
    parts = [p.lower() for p in img_path.parts]
    if "images" in parts:
        i = parts.index("images")
        new_parts = list(img_path.parts)
        new_parts[i] = "labels"
        lbl = Path(*new_parts).with_suffix(".txt")
        return lbl
    if labels_root_guess:
        return labels_root_guess / (img_path.stem + ".txt")
    return img_path.with_suffix(".txt")

def collect_images(root):
    imgs = []
    if not root or not root.exists():
        return imgs
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXTS and p.is_file():
            imgs.append(p)
    return imgs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True, help="เช่น C:\\yolo8_strip\\ultralytics\\output_k_fold2_boost")
    ap.add_argument("--folds", nargs="+", default=["0","1","2","3","4"])
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    folds = [int(f) for f in args.folds]

    for fold in folds:
        DATASET_YAML = base_dir / str(fold) / "dataset.yaml"
        OUT_DIR = base_dir / str(fold) / "_splits2"
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        with open(DATASET_YAML, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        train_dir = Path(cfg.get("train"))
        val_dir = Path(cfg.get("validation") or cfg.get("val") or "")

        if not train_dir.exists():
            raise SystemExit(f"[fold {fold}] train path not found: {train_dir}")

        labels_root_guess = None
        parts = [p.lower() for p in train_dir.parts]
        if "images" in parts:
            i = parts.index("images")
            new_parts = list(train_dir.parts)
            new_parts[i] = "labels"
            labels_root_guess = Path(*new_parts)

        candidates = collect_images(train_dir) + collect_images(val_dir)
        candidates = sorted(set(map(lambda p: p.resolve(), candidates)))

        if not candidates:
            raise SystemExit(f"[fold {fold}] No images found under train/validation dirs.")

        filtered = []
        dropped_no_label = 0
        for img in candidates:
            lbl = find_label_for_image(img, labels_root_guess)
            if lbl.exists() and lbl.is_file():
                filtered.append(str(img))
            else:
                dropped_no_label += 1

        if not filtered:
            raise SystemExit(f"[fold {fold}] No images left after checking labels. "
                             f"(dropped_no_label={dropped_no_label})")

        # สุ่มและแบ่ง
        random.shuffle(filtered)
        n = len(filtered)
        n_train = int(n * RATIOS[0])
        n_val = int(n * RATIOS[1])
        train2 = filtered[:n_train]
        val2 = filtered[n_train:n_train + n_val]
        test2 = filtered[n_train + n_val:]

        def write_list(lst, name):
            p = OUT_DIR / name
            p.write_text("\n".join(lst), encoding="utf-8")
            return str(p)

        train2_list = write_list(train2, "train2.txt")
        val2_list = write_list(val2, "val2.txt")
        test2_list = write_list(test2, "test2.txt")

        dataset2 = {
            "train": train2_list,
            "val": val2_list,
            "test": test2_list,
            "nc": cfg.get("nc"),
            "names": cfg.get("name") or cfg.get("names"),
        }
        with open(OUT_DIR / "dataset2.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(dataset2, f, sort_keys=False, allow_unicode=True)

        print(f"[fold {fold}] Done:")
        print(f"  - total images: {n} (dropped_no_label={dropped_no_label})")
        print(f"  - train2: {len(train2)}")
        print(f"  - val2:   {len(val2)}")
        print(f"  - test2:  {len(test2)}")
        print(f"  dataset2.yaml -> {OUT_DIR / 'dataset2.yaml'}")

if __name__ == "__main__":
    main()
