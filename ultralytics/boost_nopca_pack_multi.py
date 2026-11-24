# boost_nopca_pack_multi.py
import os, math, shutil, argparse
from pathlib import Path
from PIL import Image
import random

# ================== Config ==================
CELL   = 'auto'     # ขนาด tile เริ่มต้น: จำนวนเต็ม, 'auto', หรือ 'follow'
GAP    = 20         # ช่องว่างระหว่าง tile (พิกเซล)
CANVAS = 640        # ขนาดภาพปลายทาง (สี่เหลี่ยมจัตุรัส)
SPLITS = ["training", "validation", "test"]

# บังคับให้โหมด auto สร้างกริดได้อย่างน้อย MIN_GRID_RC x MIN_GRID_RC ช่อง
MIN_GRID_RC = 3     # ถ้าอยากให้ใหญ่ขึ้น/น้อยลง ปรับเป็น 2 หรือ 4 ได้

# ชื่อโฟลเดอร์ฐาน ( Base Names )
BASE_SRC_KFOLDS = ["output_k_fold", "output_k_fold2", "output_k_fold3"]

DST_KFOLDs_NOPCA = [
    "output_k_fold_boost_nopca",
    "output_k_fold_boost_nopca2",
    "output_k_fold_boost_nopca3",
]

DST_KFOLDs_PCA = [
    "output_k_fold_boost_pca",
    "output_k_fold_boost_pca2",
    "output_k_fold_boost_pca3",
]

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
# =============================================

def ensure_clean_dir(p):
    if os.path.exists(p):
        shutil.rmtree(p)
    os.makedirs(p)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def list_images(folder):
    if not os.path.isdir(folder):
        return []
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS)])

def read_first_class(label_path):
    """อ่านคลาสแรกจากไฟล์ label (YOLO txt)"""
    if not os.path.exists(label_path):
        return None
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return line.split()[0]
    return None

def read_all_boxes(label_path):
    """อ่านทุกกล่องจากไฟล์ label คืนเป็น list[str line]"""
    if not os.path.exists(label_path):
        return []
    lines = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
    return lines

def first_image_size(src_img_dir):
    """คืน (w, h) ของรูปแรกที่เปิดได้ในโฟลเดอร์"""
    for f in list_images(src_img_dir):
        p = os.path.join(src_img_dir, f)
        try:
            with Image.open(p) as im:
                return im.size
        except Exception:
            pass
    return None

def grid_params(cell, gap, canvas):
    stride = cell + gap
    usable = canvas - gap
    cols = max(1, math.floor(usable / stride))
    rows = max(1, math.floor(usable / stride))
    max_slots = cols * rows
    return stride, cols, rows, max_slots

def _auto_cell(src_img_dir, default_cell, gap, canvas):
    sz = first_image_size(src_img_dir)
    if sz is not None:
        cell = min(sz[0], sz[1])
    else:
        cell = 40 if isinstance(default_cell, str) else int(default_cell)

    max_cell_for_min_rc = (canvas - gap * (MIN_GRID_RC + 1)) // MIN_GRID_RC
    if max_cell_for_min_rc > 0:
        cell = min(cell, max_cell_for_min_rc)

    cell = max(1, int(cell))
    return cell

# ---------------- pack: train/val (multi-object) ----------------
def pack_split(
    src_img_dir, src_lbl_dir, out_img_dir, out_lbl_dir,
    cell_mode, gap=GAP, canvas=CANVAS, default_cell=CELL,
    start_idx=0, shuffle=False, seed=0
):
    ensure_dir(out_img_dir)
    ensure_dir(out_lbl_dir)

    files = list_images(src_img_dir)
    if not files:
        return 0

    if shuffle:
        rnd = random.Random(seed)
        rnd.shuffle(files)

    mode_follow = (str(cell_mode).lower() == "follow")

    if not mode_follow:
        if str(cell_mode).lower() == "auto":
            cell = _auto_cell(src_img_dir, default_cell, gap, canvas)
        else:
            try:
                cell = int(cell_mode)
            except Exception:
                cell = 40 if isinstance(default_cell, str) else int(default_cell)
        stride, cols, rows, max_slots = grid_params(cell, gap, canvas)

    canvas_idx = start_idx
    img_canvas = None
    lbl_lines = []

    x = y = 0
    row_h = 0
    slot_idx = 0

    def flush_canvas():
        nonlocal img_canvas, lbl_lines, canvas_idx, x, y, row_h, slot_idx
        if img_canvas is None:
            return
        out_name = f"{canvas_idx:06d}.jpg"
        img_canvas.save(os.path.join(out_img_dir, out_name), "JPEG")
        with open(os.path.join(out_lbl_dir, out_name.replace(".jpg", ".txt")), "w", encoding="utf-8") as f:
            f.write("\n".join(lbl_lines).strip())
        canvas_idx += 1
        img_canvas = None
        lbl_lines = []
        x = y = 0
        row_h = 0
        slot_idx = 0

    for fname in files:
        if img_canvas is None:
            img_canvas = Image.new("RGB", (canvas, canvas), (255, 255, 255))
            lbl_lines = []
            x = y = 0
            row_h = 0
            slot_idx = 0

        src_path = os.path.join(src_img_dir, fname)
        im = Image.open(src_path).convert("RGB")

        if mode_follow:
            tw, th = im.size
            if x == 0: x = gap
            if y == 0: y = gap

            if x + tw > canvas - gap + 1:
                x = gap
                y = y + row_h + gap
                row_h = 0

            if y + th > canvas - gap + 1:
                flush_canvas()
                img_canvas = Image.new("RGB", (canvas, canvas), (255, 255, 255))
                x = y = gap
                row_h = 0

            img_canvas.paste(im, (x, y))

            boxes = read_all_boxes(os.path.join(src_lbl_dir, Path(fname).stem + ".txt"))
            for b in boxes:
                parts = b.split()
                if len(parts) != 5:
                    continue
                cls, xc, yc, w, h = parts
                xc = float(xc); yc = float(yc); w = float(w); h = float(h)
                xc_canvas = (x + xc * tw) / canvas
                yc_canvas = (y + yc * th) / canvas
                w_canvas  = (w * tw) / canvas
                h_canvas  = (h * th) / canvas
                lbl_lines.append(f"{cls} {xc_canvas:.6f} {yc_canvas:.6f} {w_canvas:.6f} {h_canvas:.6f}")

            x = x + tw + gap
            row_h = max(row_h, th)

        else:
            tile = im.resize((cell, cell))

            r = slot_idx // cols
            c = slot_idx % cols
            xg = gap + c * stride
            yg = gap + r * stride

            if xg + cell > canvas - gap + 1 or yg + cell > canvas - gap + 1:
                flush_canvas()
                img_canvas = Image.new("RGB", (canvas, canvas), (255, 255, 255))
                slot_idx = 0
                r = c = 0
                xg = yg = gap

            img_canvas.paste(tile, (xg, yg))

            boxes = read_all_boxes(os.path.join(src_lbl_dir, Path(fname).stem + ".txt"))
            for b in boxes:
                parts = b.split()
                if len(parts) != 5:
                    continue
                cls, xc, yc, w, h = parts
                xc = float(xc); yc = float(yc); w = float(w); h = float(h)
                xc_canvas = (xg + xc * cell) / canvas
                yc_canvas = (yg + yc * cell) / canvas
                w_canvas  = (w * cell) / canvas
                h_canvas  = (h * cell) / canvas
                lbl_lines.append(f"{cls} {xc_canvas:.6f} {yc_canvas:.6f} {w_canvas:.6f} {h_canvas:.6f}")

            slot_idx += 1
            if slot_idx >= max_slots:
                flush_canvas()

    flush_canvas()
    return canvas_idx - start_idx

# ---------------- pack: test (single-label whole image) ----------------
def pack_split_singlelabel(
    src_img_dir, src_lbl_dir, out_img_dir, out_lbl_dir,
    cell_mode, gap=GAP, canvas=CANVAS, default_cell=CELL,
    start_idx=0, shuffle=False, seed=0
):
    ensure_dir(out_img_dir)
    ensure_dir(out_lbl_dir)

    files = list_images(src_img_dir)
    if not files:
        return 0

    if shuffle:
        rnd = random.Random(seed)
        rnd.shuffle(files)

    mode_follow = (str(cell_mode).lower() == "follow")

    if not mode_follow:
        if str(cell_mode).lower() == "auto":
            cell = _auto_cell(src_img_dir, default_cell, gap, canvas)
        else:
            try:
                cell = int(cell_mode)
            except Exception:
                cell = 40 if isinstance(default_cell, str) else int(default_cell)
        cell = max(1, int(cell))

    canvas_idx = start_idx
    for fname in files:
        src_path = os.path.join(src_img_dir, fname)
        tile = Image.open(src_path).convert("RGB")

        if mode_follow:
            tw, th = tile.size
        else:
            tile = tile.resize((cell, cell))
            tw, th = cell, cell

        img_canvas = Image.new("RGB", (canvas, canvas), (255, 255, 255))
        x = (canvas - tw) // 2
        y = (canvas - th) // 2
        img_canvas.paste(tile, (x, y))

        cls = read_first_class(os.path.join(src_lbl_dir, Path(fname).stem + ".txt")) or "0"

        xc = (x + tw / 2) / canvas
        yc = (y + th / 2) / canvas
        w  = tw / canvas
        h  = th / canvas

        out_name = f"{canvas_idx:06d}.jpg"
        img_canvas.save(os.path.join(out_img_dir, out_name), "JPEG")
        with open(os.path.join(out_lbl_dir, out_name.replace(".jpg", ".txt")), "w", encoding="utf-8") as f:
            f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        canvas_idx += 1

    return canvas_idx - start_idx

# ---------------- dataset.yaml helper ----------------
def extract_nc_names_from_yaml_text(y):
    nc_line = None
    names_line = None

    lines = [ln.strip() for ln in y.splitlines() if ln.strip()]
    for ln in lines:
        if ln.startswith("nc:"):
            nc_line = ln
        if ln.startswith("names:"):
            names_line = ln
        elif ln.startswith("name:") and names_line is None:
            names_line = ln.replace("name:", "names:", 1)

    if nc_line is None:
        nc_line = "nc: 1"
    if names_line is None:
        names_line = "names: [class0]"

    return f"{nc_line}\n{names_line}\n"

def write_dataset_yaml(src_yaml_path, out_yaml_path, train_dir, val_dir):
    tail = "nc: 1\nnames: [class0]\n"
    if os.path.exists(src_yaml_path):
        with open(src_yaml_path, "r", encoding="utf-8") as f:
            y = f.read()
        tail = extract_nc_names_from_yaml_text(y)

    new_yaml = f"train: training\nval: validation\ntest: test\n{tail.strip()}\n"
    with open(out_yaml_path, "w", encoding="utf-8") as f:
        f.write(new_yaml)

# ---------------- append single samples ----------------
def pick_representatives_per_class(src_img_dir, src_lbl_dir, per_class=1):
    picks = {}
    for fname in list_images(src_img_dir):
        cls = read_first_class(os.path.join(src_lbl_dir, Path(fname).stem + ".txt"))
        if cls is None:
            continue
        if cls not in picks:
            picks[cls] = []
        if len(picks[cls]) < per_class:
            picks[cls].append(fname)
    return picks

def append_single_samples(
    src_img_dir, src_lbl_dir,
    out_img_dir, out_lbl_dir,
    cell_mode,
    per_class=1,
):
    exist = list_images(out_img_dir)
    start_idx = len(exist)

    reps = pick_representatives_per_class(src_img_dir, src_lbl_dir, per_class)

    mode_follow = (str(cell_mode).lower() == "follow")
    if not mode_follow:
        if str(cell_mode).lower() == "auto":
            cell = _auto_cell(src_img_dir, CELL, GAP, CANVAS)
        else:
            try:
                cell = int(cell_mode)
            except:
                cell = 40
        cell = max(1, int(cell))

    idx = start_idx
    for cls, flist in reps.items():
        for fname in flist:
            src_img = os.path.join(src_img_dir, fname)
            src_lbl = os.path.join(src_lbl_dir, Path(fname).stem + ".txt")

            img = Image.open(src_img).convert("RGB")
            if not mode_follow:
                img = img.resize((cell, cell))

            out_name = f"{idx:06d}.jpg"
            img.save(os.path.join(out_img_dir, out_name), "JPEG")

            shutil.copy(
                src_lbl,
                os.path.join(out_lbl_dir, out_name.replace(".jpg", ".txt"))
            )
            idx += 1

    return idx - start_idx

def append_single_centered_samples(
    src_img_dir, src_lbl_dir,
    out_train_img_dir, out_train_lbl_dir,
    cell_mode,
    per_class=1,
    gap=GAP, canvas=CANVAS, default_cell=CELL,
):
    exist = list_images(out_train_img_dir)
    start_idx = len(exist)

    reps = pick_representatives_per_class(src_img_dir, src_lbl_dir, per_class=per_class)

    pairs = []
    for cls, flist in reps.items():
        for fname in flist:
            pairs.append((cls, fname))

    mode_follow = (str(cell_mode).lower() == "follow")
    if not mode_follow:
        if str(cell_mode).lower() == "auto":
            cell = _auto_cell(src_img_dir, default_cell, gap, canvas)
        else:
            try:
                cell = int(cell_mode)
            except Exception:
                cell = 40 if isinstance(default_cell, str) else int(default_cell)
        cell = max(1, int(cell))

    idx = start_idx
    for cls, fname in pairs:
        src_path = os.path.join(src_img_dir, fname)
        tile = Image.open(src_path).convert("RGB")

        if mode_follow:
            tw, th = tile.size
        else:
            tile = tile.resize((cell, cell))
            tw, th = cell, cell

        img_canvas = Image.new("RGB", (canvas, canvas), (255, 255, 255))
        x = (canvas - tw) // 2
        y = (canvas - th) // 2
        img_canvas.paste(tile, (x, y))

        xc = (x + tw / 2) / canvas
        yc = (y + th / 2) / canvas
        w  = tw / canvas
        h  = th / canvas

        out_name = f"{idx:06d}.jpg"
        img_canvas.save(os.path.join(out_train_img_dir, out_name), "JPEG")
        with open(os.path.join(out_train_lbl_dir, out_name.replace(".jpg", ".txt")),
                  "w", encoding="utf-8") as f:
            f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
        idx += 1

    return idx - start_idx


# ---------------- main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", default=str(CELL), help="ขนาด tile")
    parser.add_argument("--pca-mode", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=0, help="จำนวนรอบของ Repacking")
    parser.add_argument("--dest_root", type=str, default=".", help="พาธโฟลเดอร์ปลายทางหลักที่จะเซฟผลลัพธ์")
    
    # 🌟 เพิ่ม Parameter ตัวนี้เข้ามาครับ
    parser.add_argument("--dataset_name", type=str, default="", help="ชื่อของ dataset (เช่น heritage, iris)")

    args = parser.parse_args()
    cell_mode = args.cell

    if args.pca_mode:
        dst_kfolds = DST_KFOLDs_PCA
        print(f">> PCA MODE: จะเขียนไปที่โฟลเดอร์ *_boost_pca* ภายใน {args.dest_root}")
    else:
        dst_kfolds = DST_KFOLDs_NOPCA
        print(f">> NO PCA MODE: จะเขียนไปที่โฟลเดอร์ *_boost_nopca* ภายใน {args.dest_root}")

    # 🌟 สร้างชื่อโฟลเดอร์ต้นทางแบบ ไดนามิก (ถ้าระบุ --dataset_name มา จะเติม _<dataset_name> ให้)
    if args.dataset_name:
        src_kfolds = [f"{base}_{args.dataset_name}" for base in BASE_SRC_KFOLDS]
    else:
        src_kfolds = BASE_SRC_KFOLDS

    for src_folder, dst_folder_name in zip(src_kfolds, dst_kfolds):
        dst_root = os.path.join(args.dest_root, dst_folder_name)
        src_root = src_folder
        
        print(f"==> Processing {src_root} → {dst_root}  (cell={cell_mode}, K={args.k})")
        ensure_clean_dir(dst_root)

        if not os.path.isdir(src_root):
            print(f"  (skip) {src_root} not found")
            continue

        for fold in sorted(os.listdir(src_root)):
            fold_src = os.path.join(src_root, fold)
            if not os.path.isdir(fold_src):
                continue

            print(f"  Fold: {fold}")
            fold_dst = os.path.join(dst_root, fold)
            ensure_clean_dir(fold_dst)

            for sp in SPLITS:
                in_img = os.path.join(fold_src, sp, "images")
                in_lbl = os.path.join(fold_src, sp, "labels")

                out_img = os.path.join(fold_dst, sp, "images")
                out_lbl = os.path.join(fold_dst, sp, "labels")
                ensure_dir(out_img); ensure_dir(out_lbl)

                added_noncenter = 0
                added_center = 0

                if sp == "test":
                    created = pack_split_singlelabel(
                        in_img, in_lbl, out_img, out_lbl,
                        cell_mode,
                        start_idx=len(list_images(out_img)),
                        shuffle=False,
                        seed=args.seed
                    )
                else:
                    K = args.k
                    created = 0

                    created += pack_split(
                        in_img, in_lbl, out_img, out_lbl,
                        cell_mode,
                        start_idx=len(list_images(out_img)),
                        shuffle=False,
                        seed=args.seed
                    )

                    if K >= 2:
                        for i in range(1, K):
                            created += pack_split(
                                in_img, in_lbl, out_img, out_lbl,
                                cell_mode,
                                start_idx=len(list_images(out_img)),
                                shuffle=True,
                                seed=args.seed + i
                            )

                    added_noncenter = append_single_samples(
                        in_img, in_lbl, out_img, out_lbl, cell_mode, per_class=10
                    )
                    added_center = append_single_centered_samples(
                        in_img, in_lbl, out_img, out_lbl, cell_mode, per_class=10
                    )

                print(f"    Split: {sp} | created={created} | +noncenter={added_noncenter} | +center={added_center}")
                created += added_noncenter + added_center

            src_yaml = os.path.join(fold_src, "dataset.yaml")
            out_yaml = os.path.join(fold_dst, "dataset.yaml")
            write_dataset_yaml(
                src_yaml, out_yaml,
                os.path.join(dst_root, fold, "training"),
                os.path.join(dst_root, fold, "validation"),
            )

        print(f"DONE {dst_root}\n")

if __name__ == "__main__":
    main()