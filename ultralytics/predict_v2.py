# predict_v2.py
# Predict with a model trained by train_v2.py: csv rows -> strip image (same encoder as gen_folds.py) -> YOLO -> class.
# Needs only numpy + Pillow + ultralytics (no pandas / scikit-learn / lib.py), so it runs in the training venv.
#
#   python predict_v2.py --group iris_nopca_algo1_k8 --fold 0 --input new_rows.csv       console: top class + confidence per row
#   python predict_v2.py --group iris_nopca_algo1_k8 --fold 0 --input test_input.csv --no-predict      only csv -> test_input.jpg
#   python predict_v2.py --group iris_nopca_algo1_k8 --fold 0 --images predictions/images/test_input.jpg   one ready-made image
#   python predict_v2.py --group iris_nopca_algo1_k8 --fold 0 --test            the test csv of that fold (accuracy + confusion matrix)
#   python predict_v2.py --group iris_nopca_algo1_k8 --fold 0 --images out/iris/nopca/algo1/repacking_k_8/0/test/images
#                                                                                   a folder of ready-made images (labels read from ../labels if present)
#
# What it needs (all produced by the earlier steps):
#   <project_root>/<group>/train_manifest.json  -> best.pt of the fold          (train_v2.py)
#   <data_root>/<dataset>/<mode>/algo<a>/manifest.json -> Ls/Hs of the fold, feature order, class order, cell size   (gen_folds.py;
#        copied to out/ by: python transfer.py export --what sets ...)
#   --test also needs <data_root>/<dataset>/split/<mode>/fold_<f>/test.csv      (python transfer.py export --what folder --path <dataset>/split)
# roots come from train_config.json (data_root, project_root); --data-root / --project-root / --config override them.
#
# --input csv: one row per sample, columns named like the features in the manifest (extra columns are ignored).
#   The values must already be preprocessed the way the dataset loader does (e.g. automobile: label-encoded, no '?'),
#   i.e. like the columns of split/<mode>/fold_<f>/test.csv. For mode "pca" the columns must be PC1, PC2 values: the PCA
#   (fitted on the whole dataset when the split was made) is not stored, so raw features cannot be projected here.
# Output: a csv with one row per sample (predicted class, confidence, score per class, true class if known) + a summary json.
# The class of a sample = the class of its most confident detected box (the whole strip is labelled as one box).

import argparse
import csv
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import train_v2

TILE = 10          # pixels of one feature cell in the strip (lib.strip_encode1 / strips_encode2)
NORM_LIMIT = 999_999
HERE = Path(__file__).resolve().parent


# ----------------------------------------------------------------- strip encoders (same output as gen_folds / lib)
def _grid(tiles, fill):
    n = math.ceil(math.sqrt(len(tiles)))
    tiles = list(tiles) + [np.full((TILE, TILE, 3), fill, dtype=np.uint8)] * (n * n - len(tiles))
    rows = [np.concatenate(tiles[r * n:(r + 1) * n], axis=1) for r in range(n)]
    return Image.fromarray(np.concatenate(rows, axis=0))


def encode_algo1(values, Ls, Hs):
    tiles = []
    for v, L, H in zip(values, Ls, Hs):
        v = min(max(v, L), H)
        inter = 255 if H == L else ((v - L) / (H - L)) * 255
        tiles.append(np.full((TILE, TILE, 3), inter, dtype=np.uint8))
    return _grid(tiles, 0)


def _pairs(s):
    out = [s[0]] if len(s) % 2 != 0 else []
    return out + [s[i:i + 2] for i in range(len(s) % 2, len(s), 2)]


def _digits3(x):
    d = [int(p) for p in _pairs(str(abs(int(x))))]
    d = [0] * (3 - len(d)) + d
    return [-v for v in d] if x < 0 else d


def _scale(L, H, v):  # lib.interpolate_value
    if H == L:
        return 0
    v = min(max(v, L), H)
    return ((v - L) * 240 / (H - L)) + 10


def encode_algo2(values, Ls, Hs):
    lo, hi = _digits3(-NORM_LIMIT), _digits3(NORM_LIMIT)
    tiles = []
    for v, L, H in zip(values, Ls, Hs):
        if H == L:
            norm = 0.0
        else:
            n = min(max(2 * ((v - L) / (H - L)) - 1, -1.0), 1.0)
            norm = float(np.round(n * NORM_LIMIT))
        d = _digits3(int(min(max(norm, -NORM_LIMIT), NORM_LIMIT)))
        rgb = [round(_scale(lo[j], hi[j], d[j])) for j in range(3)]
        t = np.zeros((TILE, TILE, 3), dtype=np.uint8)
        t[:, :] = rgb
        tiles.append(t)
    return _grid(tiles, 255)


ENCODERS = {1: encode_algo1, 2: encode_algo2}


def _jpeg_bytes(img):
    buf = io.BytesIO()
    img.save(buf, "JPEG")
    return buf.getvalue()


def _jpeg(img):
    return Image.open(io.BytesIO(_jpeg_bytes(img))).convert("RGB")


def make_canvas(values, algo, Ls, Hs, cell, canvas, roundtrip):
    """one sample -> (decoded canvas image, the jpeg bytes it came from), built like the test images of gen_folds.py.
    Write the bytes to disk (not the image): re-encoding the decoded image would compress it a second time."""
    strip = ENCODERS[algo](values, Ls, Hs)
    if roundtrip:
        strip = _jpeg(strip)
    img = Image.new("RGB", (canvas, canvas), (255, 255, 255))
    x = (canvas - cell) // 2
    img.paste(strip.convert("RGB").resize((cell, cell)), (x, x))
    data = _jpeg_bytes(img)  # gen_folds saves the canvas as .jpg
    return Image.open(io.BytesIO(data)).convert("RGB"), data


# ----------------------------------------------------------------------------------------------------- helpers
def load_json(path):
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except OSError:
        return None


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"{path} has no rows")
    return rows


def to_float(row, col, idx):
    try:
        v = float(row[col])
    except (KeyError, TypeError, ValueError):
        sys.exit(f"row {idx + 1}: column '{col}' is not a number ({row.get(col)!r})")
    if not math.isfinite(v):
        sys.exit(f"row {idx + 1}: column '{col}' is not finite ({row[col]!r})")
    return v


def best_pt(gdir, entry, fold):
    rel = (entry or {}).get("best_pt")
    for cand in ([gdir / rel.replace("\\", "/")] if rel else []) + [gdir / f"fold_{fold}" / "train" / "weights" / "best.pt"]:
        if cand.is_file():
            return cand
    sys.exit(f"best.pt of fold {fold} not found under {gdir} (train it: python train_v2.py --only {gdir.name} --folds {fold})")


def class_scores(result, n_classes):
    scores = [0.0] * n_classes
    boxes = result.boxes
    if boxes is not None and len(boxes):
        for c, p in zip(boxes.cls.tolist(), boxes.conf.tolist()):
            if int(c) < n_classes:
                scores[int(c)] = max(scores[int(c)], float(p))
    return scores


def true_from_label_file(img_path):
    p = Path(img_path)
    lbl = p.parent.parent / "labels" / (p.stem + ".txt")
    try:
        with open(lbl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    return int(line.split()[0])
    except OSError:
        pass
    return None


class Ctx:
    """everything needed to turn csv rows into images and to find the model (read once, reused by predict_watch_v2.py)"""


def load_context(group, fold, config=None, data_root=None, project_root=None, weights=None):
    cfg = train_v2.load_config(config or str(train_v2.DEFAULT_CONFIG))
    c = Ctx()
    c.data_root = train_v2.resolve(data_root or cfg["data_root"])
    c.gdir = train_v2.resolve(project_root or cfg["project_root"]) / group
    tm = load_json(c.gdir / train_v2.MANIFEST)
    if not tm:
        sys.exit(f"no training manifest: {c.gdir / train_v2.MANIFEST} (train first: python train_v2.py)")
    c.group, c.fold = group, str(fold)
    c.ds, c.mode, c.algo, c.k = tm["dataset"], tm["mode"], int(tm["algo"]), str(tm["k"])
    c.entry = tm["folds"].get(c.fold)
    c.weights = Path(weights) if weights else None
    if c.entry and c.entry.get("status") != "done" and not weights:
        print(f"warning: fold {c.fold} status is '{c.entry.get('status')}' (not done)", file=sys.stderr)

    am_path = c.data_root / c.ds / c.mode / f"algo{c.algo}" / "manifest.json"
    am = load_json(am_path)
    if not am:
        sys.exit(f"{am_path} not found (python transfer.py export --what sets --datasets {c.ds} --modes {c.mode} "
                 f"--algos {c.algo} --k_list {c.k})")
    if tm.get("split_id") and am.get("split_id") != tm["split_id"]:
        sys.exit(f"split_id differs: model was trained on {tm['split_id']}, manifest says {am.get('split_id')}")
    if c.fold not in am["folds"]:
        sys.exit(f"fold {c.fold} not in the manifest (folds: {', '.join(am['folds'])})")
    c.feats, c.labels = am["features"], [str(x) for x in am["labels"]]
    c.Ls, c.Hs = am["folds"][c.fold]["Ls"], am["folds"][c.fold]["Hs"]
    run = am["k_runs"].get(c.k)
    if not run:
        sys.exit(f"k={c.k} is not in {am_path} (k_runs: {', '.join(am['k_runs'])})")
    c.cell = run["counts"][c.fold]["test"]["cell"]
    c.canvas, c.roundtrip = int(run["canvas"]), bool(run["jpeg_roundtrip"])
    try:
        import PIL
        pil_gen = am.get("versions", {}).get("pillow")
        if pil_gen and pil_gen.split(".")[:2] != PIL.__version__.split(".")[:2]:
            print(f"warning: images were generated with Pillow {pil_gen}, this python has {PIL.__version__}: "
                  f"JPEG/resize can differ slightly from the training images", file=sys.stderr)
    except Exception:
        pass
    c.label_col = (load_json(c.data_root / c.ds / "split" / "manifest.json") or {}).get("label_header")
    return c


def resolve_weights(c):
    return c.weights or best_pt(c.gdir, c.entry, c.fold)


def rows_to_items(c, rows, source, label_col=None, test=False, where="csv"):
    """csv rows -> ([(name, PIL image, true class index or None, extra fields)], {name: jpeg bytes})"""
    missing = [f for f in c.feats if f not in rows[0]]
    if missing:
        sys.exit(f"{where} lacks feature column(s): {', '.join(missing)}\nexpected (in this order): {', '.join(c.feats)}")
    # gen_folds numbers the test images in the order of sorted(f"{row_id}.jpg"); same order here for --test
    order = sorted(range(len(rows)), key=lambda i: f"{rows[i]['row_id']}.jpg") if test and "row_id" in rows[0] else range(len(rows))
    items, blobs = [], {}
    for n, i in enumerate(order):
        r = rows[i]
        img, blob = make_canvas([to_float(r, f, i) for f in c.feats], c.algo, c.Ls, c.Hs, c.cell, c.canvas, c.roundtrip)
        truth = None
        if label_col and label_col in r:
            truth = c.labels.index(str(r[label_col])) if str(r[label_col]) in c.labels else -1
        extra = {"row_id": r["row_id"]} if "row_id" in r else {"row": i + 1}
        if test:
            name = f"{n:06d}.jpg"  # same numbering as the test images of gen_folds.py
        else:
            name = f"{source}.jpg" if len(rows) == 1 else f"{source}_{n + 1:04d}.jpg"
        items.append((name, img, truth, extra))
        blobs[name] = blob
    return items, blobs


# --------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", required=True, help="<dataset>_<mode>_algo<a>_k<k>, e.g. iris_nopca_algo1_k8")
    ap.add_argument("--fold", type=int, default=0)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="csv with the feature columns")
    src.add_argument("--test", action="store_true", help="the test csv of the fold (from the split)")
    src.add_argument("--images", help="one image or a folder of ready-made canvas images")
    ap.add_argument("--label-col", default=None, help="column with the true class (default: taken from the split manifest if present)")
    ap.add_argument("--out", default=None, help="result csv (default: predictions/<group>_fold<f>_<source>.csv)")
    ap.add_argument("--save-images", default=None, help="also write the generated images to this folder (named <csv name>.jpg)")
    ap.add_argument("--no-predict", action="store_true", help="only create the image(s) from the csv (default folder: predictions/images), no model")
    ap.add_argument("--show", type=int, default=20, help="print one line per sample on the console up to this many samples")
    ap.add_argument("--weights", default=None, help="use this .pt instead of the best.pt of the fold")
    ap.add_argument("--conf", type=float, default=0.001, help="lowest box confidence kept (low on purpose: the top box decides)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--config", default=str(train_v2.DEFAULT_CONFIG))
    ap.add_argument("--data-root", dest="data_root", default=None)
    ap.add_argument("--project-root", dest="project_root", default=None)
    a = ap.parse_args()

    ctx = load_context(a.group, a.fold, a.config, a.data_root, a.project_root, a.weights)
    ds, mode, fold, labels, canvas, cell = ctx.ds, ctx.mode, ctx.fold, ctx.labels, ctx.canvas, ctx.cell
    label_col = a.label_col or ctx.label_col

    items = []  # (name, PIL image, true class index or None, extra csv fields)
    blobs = {}  # name -> jpeg bytes of the generated images
    if a.images:
        ip = Path(a.images)
        files = [ip] if ip.is_file() else sorted(p for p in ip.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if not files:
            sys.exit(f"no images in {a.images}")
        for p in files:
            items.append((p.name, Image.open(p).convert("RGB"), true_from_label_file(p), {}))
        source = ip.stem if ip.is_file() else "images"
    else:
        if a.test:
            csv_path = ctx.data_root / ds / "split" / mode / f"fold_{fold}" / "test.csv"
            if not csv_path.is_file():
                sys.exit(f"{csv_path} not found (python transfer.py export --what folder --path {ds}/split)")
            source = "test"
        else:
            csv_path, source = Path(a.input), Path(a.input).stem
        items, blobs = rows_to_items(ctx, read_csv(csv_path), source, label_col, a.test, str(csv_path))
        if a.save_images or a.no_predict:
            save_dir = Path(a.save_images) if a.save_images else HERE / "predictions" / "images"
            save_dir.mkdir(parents=True, exist_ok=True)
            for name, blob in blobs.items():
                (save_dir / name).write_bytes(blob)
            print(f"{len(items)} image(s) written to {save_dir}")
    if a.no_predict:
        return

    from ultralytics import YOLO
    weights = resolve_weights(ctx)
    model = YOLO(str(weights))
    names = model.names
    if len(names) != len(labels):
        print(f"warning: model has {len(names)} classes, manifest has {len(labels)}", file=sys.stderr)
    kw = {"imgsz": canvas, "conf": a.conf, "verbose": False}
    if a.device is not None:
        kw["device"] = a.device

    preds = []
    for s in range(0, len(items), a.batch):
        chunk = [im for _, im, _, _ in items[s:s + a.batch]]
        preds += [class_scores(r, len(labels)) for r in model.predict(chunk, **kw)]

    out = Path(a.out) if a.out else HERE / "predictions" / f"{a.group}_fold{fold}_{source}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    have_truth = any(t is not None for _, _, t, _ in items)
    n_ok = n_known = n_none = 0
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        extra_cols = list(items[0][3])
        w.writerow(["image", *extra_cols, "pred", "conf", *[f"score_{c}" for c in labels]] + (["true", "correct"] if have_truth else []))
        for (name, _, truth, extra), sc in zip(items, preds):
            top = int(np.argmax(sc)) if max(sc) > 0 else None
            n_none += top is None
            row = [name, *[extra[c] for c in extra_cols], labels[top] if top is not None else "none",
                   f"{max(sc):.4f}", *[f"{x:.4f}" for x in sc]]
            if have_truth:
                ok = top is not None and truth is not None and truth >= 0 and top == truth
                row += [labels[truth] if truth is not None and truth >= 0 else ("?" if truth is not None else ""),
                        "" if truth is None else int(ok)]
                if truth is not None and truth >= 0:
                    n_known += 1
                    n_ok += ok
                    if top is not None:
                        cm[truth, top] += 1
            w.writerow(row)

    summary = {"group": a.group, "fold": a.fold, "source": source, "weights": str(weights), "samples": len(items),
               "no_detection": n_none, "cell": cell, "canvas": canvas, "out_csv": str(out)}
    if len(items) <= a.show:
        print()
        for (name, _, truth, _), sc in zip(items, preds):
            order = sorted(range(len(labels)), key=lambda j: -sc[j])
            if sc[order[0]] <= 0:
                print(f"{name}: no detection (nothing above conf {a.conf})")
                continue
            rest = ", ".join(f"{labels[j]} {sc[j]:.3f}" for j in order[1:3] if sc[j] > 0)
            real = f"   [true: {labels[truth]}]" if truth is not None and truth >= 0 else ""
            print(f"{name}: {labels[order[0]]}  (confidence {sc[order[0]]:.3f})" + (f"   others: {rest}" if rest else "") + real)
        print()
    print(f"{len(items)} sample(s) -> {out}")
    if have_truth and n_known:
        summary.update(accuracy=round(n_ok / n_known, 4), correct=int(n_ok), with_truth=n_known)
        print(f"accuracy {n_ok}/{n_known} = {n_ok / n_known:.4f}   (no detection: {n_none})")
        print("confusion matrix (rows = true, columns = predicted; samples with no detection are not counted):")
        w0 = max(len(c) for c in labels) + 1
        print(" " * w0 + "".join(f"{c[:8]:>9}" for c in labels))
        for i, c in enumerate(labels):
            print(f"{c:<{w0}}" + "".join(f"{cm[i, j]:>9}" for j in range(len(labels))))
        summary["confusion_matrix"] = {"labels": labels, "matrix": cm.tolist()}
    else:
        print("class counts: " + ", ".join(f"{c}={sum(1 for sc in preds if max(sc) > 0 and int(np.argmax(sc)) == i)}" for i, c in enumerate(labels))
              + f", none={n_none}")
    with open(out.with_suffix(".summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
