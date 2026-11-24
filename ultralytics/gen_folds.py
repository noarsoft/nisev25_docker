# gen_folds.py
# Generate k-fold strip datasets (train/val/test) without writing per-image strips.
#
# Step 1 - split: write per-fold CSVs + split/manifest.json
#   python gen_folds.py split --datasets iris --dataset_root DATA --out_dir OUT --seed 42
#   python gen_folds.py split --datasets iris --dataset_root DATA --out_dir OUT --from_json_dir .   (reuse old output_kfold_data_<ds>.json)
#
# Step 2 - gen: build strip tiles in memory from the CSVs and pack them (algo 1 = grey, algo 2 = colour)
#   python gen_folds.py gen --datasets iris --out_dir OUT --modes nopca,pca --algos 1,2 --k_list 0,2,4
#
# Layout (OUT/<dataset>/):
#   split/manifest.json, split/<mode>/fold_<i>/{train,val,test}.csv
#   <mode>/algo<a>/manifest.json, <mode>/algo<a>/repacking_k_<k>/<fold>/{training,validation,test}/{images,labels}, dataset.yaml
#   <mode>/algo<a>/repacking_k_<k>/<fold>/index/{training,validation,test}.csv   which split row (row_id) every box comes from
#   <mode>/algo<a>/repacking_k_<k>/<fold>/preview/<part>/   (only with --preview) images with boxes + row_ids drawn on

import argparse
import contextlib
import csv
import hashlib
import io
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import lib

CANVAS = 640
GAP = 20
MIN_GRID_RC = 3
PER_CLASS = 10
NORM_LIMIT = 999_999
PART_FILES = ("train", "val", "test")
SPLIT_DIRS = {"train": "training", "val": "validation", "test": "test"}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_obj(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


def versions():
    return {"python": sys.version.split()[0], "numpy": np.__version__,
            "pandas": pd.__version__, "pillow": Image.__version__}


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ----------------------------------------------------------------- split
def load_dataset(dataset, dataset_dir):
    import strip_pre2  # reuse its dataset registry (CSV path, header, label)
    with quiet():
        raw_csv, header, label, pca_csv, header_pca = strip_pre2.select_dataset(dataset, str(dataset_dir))
    if not isinstance(label, str):
        raise SystemExit(f"{dataset}: multi-label datasets are not supported by gen_folds.py")
    return raw_csv, header, label, pca_csv, header_pca


def cmd_split(args):
    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        raw_csv, header, label, pca_csv, header_pca = load_dataset(ds, Path(args.dataset_root) / ds)
        out = Path(args.out_dir) / ds / "split"
        if out.exists():
            if not args.overwrite:
                raise SystemExit(f"{out} already exists (use --overwrite to replace it)")
            shutil.rmtree(out)

        if args.from_json_dir:
            json_path = Path(args.from_json_dir) / f"output_kfold_data_{ds}.json"
            with open(json_path, "r") as f:
                saved = json.load(f)
            # replay exactly the way strip_pre2.py does: it is what the old strips were built from
            with quiet():
                res = lib.preprocess_create_k_fold(raw_csv, header, label, saved[6], saved[7])
            origin = {"type": "json_replay", "file": json_path.name, "sha256": sha256_file(json_path)}
        else:
            np.random.seed(args.seed)
            with quiet():
                res = lib.preprocess_create_k_fold(raw_csv, header, label)
            origin = {"type": "random", "seed": args.seed}

        folds = [{"train": [int(x) for x in t], "val": [int(x) for x in v], "test": [int(x) for x in te]}
                 for t, v, te in zip(res[1], res[2], res[3])]

        df_raw = pd.read_csv(raw_csv)
        df_raw[label] = df_raw[label].astype(str)
        df_pca = pd.read_csv(pca_csv)
        df_pca[label] = df_pca[label].astype(str)
        if len(df_pca) != len(df_raw):
            raise SystemExit(f"{ds}: pca csv has {len(df_pca)} rows but raw csv has {len(df_raw)}")

        modes = {}
        for mode, df, feats, src in (("nopca", df_raw, header, raw_csv), ("pca", df_pca, header_pca, pca_csv)):
            modes[mode] = {"features": list(feats), "source_csv": Path(src).name, "source_sha256": sha256_file(src)}
            for i, fold in enumerate(folds):
                for part in PART_FILES:
                    sub = df.iloc[fold[part]][list(feats) + [label]].copy()
                    sub.insert(0, "row_id", fold[part])
                    path = out / mode / f"fold_{i}" / f"{part}.csv"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    sub.to_csv(path, index=False)

        manifest = {
            "dataset": ds,
            "origin": origin,
            "n_rows": len(df_raw),
            "n_folds": len(folds),
            "label_header": label,
            "labels": sorted(set(df_raw[label])),
            "modes": modes,
            "folds": [{"fold": i, "n_train": len(f["train"]), "n_val": len(f["val"]), "n_test": len(f["test"])}
                      for i, f in enumerate(folds)],
            "split_id": sha256_obj({"raw": modes["nopca"]["source_sha256"], "folds": folds})[:16],
            "versions": versions(),
        }
        write_json(out / "manifest.json", manifest)
        print(f"[{ds}] split written to {out}  (split_id={manifest['split_id']}, origin={origin['type']})")


# ------------------------------------------------------------- encoders
def encode_row_algo1(values, Ls, Hs):
    return lib.strip_encode(values, Ls, Hs)


def encode_row_algo2(values, Ls, Hs, size=10):
    # mirrors lib.strips_encode2 for a single row
    digits_min = lib.create_feature_for_strip_color(str(-NORM_LIMIT))
    digits_max = lib.create_feature_for_strip_color(str(NORM_LIMIT))
    tiles = []
    for v, L, H in zip(values, Ls, Hs):
        L = float(L)
        H = float(H)
        if H == L:
            norm = 0.0
        else:
            n = 2 * ((float(v) - L) / (H - L)) - 1
            n = min(max(n, -1.0), 1.0)
            norm = float(np.round(n * NORM_LIMIT))
        x = lib.safe_value_for_strip(norm, limit=NORM_LIMIT)
        digits = lib.create_feature_for_strip_color(str(int(x)))
        d = [round(lib.interpolate_value(digits_min[j], digits_max[j], digits[j])) for j in range(3)]
        img = Image.new("RGB", (size, size), (d[0], d[1], d[2]))
        if d[2] > 100:
            lib.insert_text_to_img(img, str(digits[2]), (0, 0, 0))
        else:
            lib.insert_text_to_img(img, str(digits[2]))
        tiles.append(img)
    fill = Image.new("RGB", (size, size), (255, 255, 255))
    matrix = lib.list_to_square_matrix(tiles, fill)
    lines = [lib.concatinate_list_img_left2right(line) for line in matrix]
    return lib.concatinate_list_img_top2bot(lines)


ENCODERS = {1: encode_row_algo1, 2: encode_row_algo2}


def build_tiles(df, feats, label, Ls, Hs, algo, roundtrip):
    """row_id -> (RGB tile, class string); roundtrip=True mimics the old save-as-.jpg / re-open step."""
    encode = ENCODERS[algo]
    tiles = {}
    sub = df[list(feats) + [label]]
    for rid, (_, row) in zip(df["row_id"], sub.iterrows()):
        img = encode([row[h] for h in feats], Ls, Hs)
        if roundtrip:
            buf = io.BytesIO()
            img.save(buf, "JPEG")
            buf.seek(0)
            img = Image.open(buf)
        tiles[int(rid)] = (img.convert("RGB"), str(row[label]))
    return tiles


# ----------------------------------------------------------------- pack
INDEX_COLS = ["image", "line", "row_id", "class", "cx", "cy", "w", "h", "kind"]


class Writer:
    """Writes images/ + labels/ of one part. Besides, <fold>/index/<part>.csv tells which split row every box comes from
    (kind: pack | single | centered | test; line = 1-based line of the box in the label file) and, with preview,
    <fold>/preview/<part>/ holds copies with the boxes and row_ids drawn on. Both are outside <part>/ so YOLO never reads them."""

    def __init__(self, split_dir, preview=False):
        self.img_dir = split_dir / "images"
        self.lbl_dir = split_dir / "labels"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.lbl_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = split_dir.parent / "index" / f"{split_dir.name}.csv"
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.preview_dir = None
        if preview:
            self.preview_dir = split_dir.parent / "preview" / split_dir.name
            self.preview_dir.mkdir(parents=True, exist_ok=True)
        self.rows = []
        self.n = 0

    def save(self, img, label_text, row_ids, kind):
        name = f"{self.n:06d}"
        img.save(self.img_dir / f"{name}.jpg", "JPEG")
        with open(self.lbl_dir / f"{name}.txt", "w", encoding="utf-8") as f:
            f.write(label_text)
        boxes = []
        for line_no, line in enumerate(label_text.strip().splitlines(), 1):
            *cls, cx, cy, w, h = line.rsplit(None, 4)
            cls = " ".join(cls)
            rid = row_ids[line_no - 1]
            boxes.append((rid, float(cx), float(cy), float(w), float(h)))
            self.rows.append([f"{name}.jpg", line_no, rid, cls, cx, cy, w, h, kind])
        if self.preview_dir is not None:
            self._preview(img, name, boxes)
        self.n += 1

    def _preview(self, img, name, boxes):
        im = img.convert("RGB")
        draw = ImageDraw.Draw(im)
        W, H = im.size
        for rid, cx, cy, w, h in boxes:
            x0, y0, x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H
            draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0))
            below = (x1 - x0) < 0.5 * W and y1 + 12 <= H  # small boxes: write the id in the white gap under the box
            draw.text((x0, y1 + 1) if below else (x0 + 2, y0 + 2), str(rid), fill=(255, 0, 0))
        im.save(self.preview_dir / f"{name}.jpg", "JPEG", quality=85)

    def close(self):
        with open(self.index_path, "w", encoding="utf-8", newline="") as f:
            csv.writer(f, lineterminator="\n").writerows([INDEX_COLS, *self.rows])


def pick_cell(cell_arg, sample_img):
    if str(cell_arg).lower() == "follow":
        raise SystemExit("--cell follow is not supported by gen_folds.py")
    if str(cell_arg).lower() == "auto":
        cell = min(sample_img.size)
        limit = (CANVAS - GAP * (MIN_GRID_RC + 1)) // MIN_GRID_RC
        if limit > 0:
            cell = min(cell, limit)
    else:
        cell = int(cell_arg)
    return max(1, int(cell))


def pack_pass(items, cell, writer):
    stride = cell + GAP
    cols = max(1, (CANVAS - GAP) // stride)
    max_slots = cols * cols
    canvas = None
    lines = []
    slot = 0
    for img, cls, rid in items:
        if canvas is None:
            canvas = Image.new("RGB", (CANVAS, CANVAS), (255, 255, 255))
            lines = []
            rids = []
            slot = 0
        r, c = divmod(slot, cols)
        xg = GAP + c * stride
        yg = GAP + r * stride
        canvas.paste(img.resize((cell, cell)), (xg, yg))
        lines.append(f"{cls} {(xg + 0.5 * cell) / CANVAS:.6f} {(yg + 0.5 * cell) / CANVAS:.6f} "
                     f"{(1.0 * cell) / CANVAS:.6f} {(1.0 * cell) / CANVAS:.6f}")
        rids.append(rid)
        slot += 1
        if slot >= max_slots:
            writer.save(canvas, "\n".join(lines).strip(), rids, "pack")
            canvas = None
    if canvas is not None:
        writer.save(canvas, "\n".join(lines).strip(), rids, "pack")


def centered_canvas(img, cls, cell):
    tile = img.resize((cell, cell))
    canvas = Image.new("RGB", (CANVAS, CANVAS), (255, 255, 255))
    x = (CANVAS - cell) // 2
    y = (CANVAS - cell) // 2
    canvas.paste(tile, (x, y))
    label = (f"{cls} {(x + cell / 2) / CANVAS:.6f} {(y + cell / 2) / CANVAS:.6f} "
             f"{cell / CANVAS:.6f} {cell / CANVAS:.6f}\n")
    return canvas, label


def representatives(items, per_class):
    picks = {}
    for img, cls, rid in items:
        lst = picks.setdefault(cls, [])
        if len(lst) < per_class:
            lst.append((img, rid))
    return picks


def write_train_like(items, cell, k, seed, split_dir, preview=False):
    # items must already be in file-name order, like the old sorted(os.listdir(...))
    w = Writer(split_dir, preview)
    pack_pass(items, cell, w)
    for i in range(1, k):
        shuffled = list(items)
        random.Random(seed + i).shuffle(shuffled)
        pack_pass(shuffled, cell, w)
    picks = representatives(items, PER_CLASS)
    for cls, imgs in picks.items():
        for img, rid in imgs:
            w.save(img.resize((cell, cell)), f"{cls} 0.5 0.5 1 1", [rid], "single")
    for cls, imgs in picks.items():
        for img, rid in imgs:
            w.save(*centered_canvas(img, cls, cell), [rid], "centered")
    w.close()
    return w.n


def write_test(items, cell, split_dir, preview=False):
    w = Writer(split_dir, preview)
    for img, cls, rid in items:
        w.save(*centered_canvas(img, cls, cell), [rid], "test")
    w.close()
    return w.n


def ordered(tiles):
    # same order as sorted(os.listdir) over "<row_id>.jpg"; every item is (tile, class, row_id)
    return [(*tiles[r], r) for r in sorted(tiles, key=lambda r: f"{r}.jpg")]


# ------------------------------------------------------------------ gen
def cmd_gen(args):
    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        root = Path(args.out_dir) / ds
        with open(root / "split" / "manifest.json", "r", encoding="utf-8") as f:
            sm = json.load(f)
        labels = sm["labels"]
        label_hdr = sm["label_header"]
        label_idx = {name: i for i, name in enumerate(labels)}
        yaml_tail = f"nc: {len(labels)}\nnames: {labels}\n"

        for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
            feats = sm["modes"][mode]["features"]
            for algo in [int(a) for a in args.algos.split(",") if a.strip()]:
                base = root / mode / f"algo{algo}"
                mpath = base / "manifest.json"
                manifest = {"dataset": ds, "mode": mode, "algo": algo, "split_id": sm["split_id"],
                            "features": feats, "labels": labels, "folds": {}, "k_runs": {}, "versions": versions()}
                if mpath.exists():
                    with open(mpath, "r", encoding="utf-8") as f:
                        old = json.load(f)
                    if old.get("split_id") == sm["split_id"]:
                        manifest["k_runs"] = old.get("k_runs", {})

                fold_data = {}
                for i in range(sm["n_folds"]):
                    fdir = root / "split" / mode / f"fold_{i}"
                    parts = {p: pd.read_csv(fdir / f"{p}.csv", dtype={label_hdr: str}) for p in PART_FILES}
                    train = parts["train"]
                    raw_l = [train[c].min() for c in feats]
                    raw_h = [train[c].max() for c in feats]
                    Ls, Hs = lib.max_min_range_expand(raw_l, raw_h)
                    manifest["folds"][str(i)] = {
                        "raw_min": [float(x) for x in raw_l], "raw_max": [float(x) for x in raw_h],
                        "Ls": [float(x) for x in Ls], "Hs": [float(x) for x in Hs],
                        "n_train": len(parts["train"]), "n_val": len(parts["val"]), "n_test": len(parts["test"]),
                    }
                    fold_data[i] = {
                        p: {rid: (img, str(label_idx[cls])) for rid, (img, cls) in
                            build_tiles(parts[p], feats, label_hdr, Ls, Hs, algo, not args.no_jpeg_roundtrip).items()}
                        for p in PART_FILES
                    }
                    print(f"[{ds}/{mode}/algo{algo}] fold {i}: tiles built")

                for k in [int(x) for x in args.k_list.split(",") if x.strip()]:
                    kdir = base / f"repacking_k_{k}"
                    if kdir.exists():
                        shutil.rmtree(kdir)
                    counts = {}
                    for i, data in fold_data.items():
                        fdst = kdir / str(i)
                        counts[str(i)] = {}
                        for p in PART_FILES:
                            items = ordered(data[p])
                            cell = pick_cell(args.cell, items[0][0])
                            if p == "test":
                                n = write_test(items, cell, fdst / SPLIT_DIRS[p], args.preview)
                            else:
                                n = write_train_like(items, cell, k, args.seed, fdst / SPLIT_DIRS[p], args.preview)
                            counts[str(i)][SPLIT_DIRS[p]] = {"images": n, "cell": cell}
                        with open(fdst / "dataset.yaml", "w", encoding="utf-8") as f:
                            f.write(f"train: training\nval: validation\ntest: test\n{yaml_tail}")
                    manifest["k_runs"][str(k)] = {
                        "k": k, "seed": args.seed, "cell": args.cell, "canvas": CANVAS, "gap": GAP,
                        "per_class_singles": PER_CLASS, "jpeg_roundtrip": not args.no_jpeg_roundtrip,
                        "counts": counts,
                    }
                    print(f"[{ds}/{mode}/algo{algo}] k={k} written to {kdir}")
                write_json(mpath, manifest)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("split", help="write per-fold train/val/test CSVs + manifest")
    sp.add_argument("--datasets", required=True, help="comma-separated names known by strip_pre2.select_dataset")
    sp.add_argument("--dataset_root", required=True, help="folder that holds <dataset>/<dataset>.csv")
    sp.add_argument("--out_dir", required=True)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--from_json_dir", default=None,
                    help="folder with old output_kfold_data_<dataset>.json; replays that split instead of a new random one")
    sp.add_argument("--overwrite", action="store_true")
    sp.set_defaults(func=cmd_split)

    gp = sub.add_parser("gen", help="build packed train/val/test images from the split CSVs")
    gp.add_argument("--datasets", required=True)
    gp.add_argument("--out_dir", required=True)
    gp.add_argument("--modes", default="nopca,pca")
    gp.add_argument("--algos", default="1,2")
    gp.add_argument("--k_list", default="0")
    gp.add_argument("--cell", default="auto")
    gp.add_argument("--seed", type=int, default=0, help="base seed of the repack shuffles (seed + i)")
    gp.add_argument("--no_jpeg_roundtrip", action="store_true",
                    help="skip the JPEG encode/decode of each tile (old pipeline saved every strip as .jpg)")
    gp.add_argument("--preview", action="store_true",
                    help="also write <fold>/preview/<part>/ : copies of the images with every box and its row_id drawn on (not used by YOLO)")
    gp.set_defaults(func=cmd_gen)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
