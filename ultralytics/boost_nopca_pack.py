# boost_nopca_pack_multi.py
import os, math, shutil
from pathlib import Path
from PIL import Image

# ================== Config ==================
CELL   = 40         # ขนาด tile
GAP    = 20         # ช่องว่างระหว่าง tile
CANVAS = 640        # ขนาดภาพปลายทาง
SPLITS = ["training", "validation", "test"]

SRC_KFOLDs = ["output_k_fold", "output_k_fold2", "output_k_fold3"]
DST_KFOLDs = ["output_k_fold_boost_nopca",
              "output_k_fold_boost_nopca2",
              "output_k_fold_boost_nopca3"]

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

def grid_params():
    """คำนวณจำนวนช่องที่วางได้บน 640x640 ตาม CELL,GAP"""
    stride = CELL + GAP
    cols = math.floor((CANVAS - GAP - CELL) / stride) + 1
    rows = cols
    max_slots = rows * cols
    return stride, cols, rows, max_slots

# ---------------- pack: train/val (multi-object) ----------------
def pack_split(src_img_dir, src_lbl_dir, out_img_dir, out_lbl_dir):
    """
    รวมหลายภาพ (tile) ลงบนแคนวาส 640x640
    labels: รวมกล่องทุกกล่องแล้วปรับเป็นพิกัดของตำแหน่ง tile
    """
    ensure_dir(out_img_dir)
    ensure_dir(out_lbl_dir)

    files = list_images(src_img_dir)
    if not files:
        return 0

    stride, cols, rows, max_slots = grid_params()

    canvas_idx = 0
    slot_idx = 0
    img_canvas = None
    lbl_lines = []  # สะสมกล่องทั้งหมดใน canvas ปัจจุบัน

    def flush_canvas():
        nonlocal img_canvas, lbl_lines, canvas_idx
        if img_canvas is None:
            return
        out_name = f"{canvas_idx:06d}.jpg"
        img_canvas.save(os.path.join(out_img_dir, out_name), "JPEG")
        with open(os.path.join(out_lbl_dir, out_name.replace(".jpg", ".txt")), "w", encoding="utf-8") as f:
            f.write("\n".join(lbl_lines).strip())
        canvas_idx += 1
        img_canvas = None
        lbl_lines = []

    for fname in files:
        if img_canvas is None:
            img_canvas = Image.new("RGB", (CANVAS, CANVAS), (255, 255, 255))
            slot_idx = 0
            lbl_lines = []

        src_path = os.path.join(src_img_dir, fname)
        tile = Image.open(src_path).convert("RGB").resize((CELL, CELL))

        # ตำแหน่งลงบนกริด
        r = slot_idx // cols
        c = slot_idx % cols
        x = GAP + c * stride
        y = GAP + r * stride

        # ถ้าเกินขอบ แสดงว่าเต็มแล้ว → flush แล้วเริ่ม canvas ใหม่
        if x + CELL > CANVAS - GAP + 1 or y + CELL > CANVAS - GAP + 1:
            flush_canvas()
            img_canvas = Image.new("RGB", (CANVAS, CANVAS), (255, 255, 255))
            slot_idx = 0
            lbl_lines = []
            r = c = 0
            x = GAP
            y = GAP

        # วางรูป
        img_canvas.paste(tile, (x, y))

        # รวม labels ของรูปนี้ แปลง bbox เป็นพิกัด canvas
        boxes = read_all_boxes(os.path.join(src_lbl_dir, Path(fname).stem + ".txt"))
        for b in boxes:
            parts = b.split()
            if len(parts) != 5:
                continue
            cls, xc, yc, w, h = parts
            xc = float(xc); yc = float(yc); w = float(w); h = float(h)
            # แปลงจากพิกัด tile (normalized 0..1) → พิกัด canvas (normalized 0..1)
            # หา center ของ tile บน canvas:
            # จุดซ้ายบน tile = (x, y), ขนาด tile = CELL, ขนาด canvas = CANVAS
            xc_canvas = (x + xc * CELL) / CANVAS
            yc_canvas = (y + yc * CELL) / CANVAS
            w_canvas  = (w * CELL) / CANVAS
            h_canvas  = (h * CELL) / CANVAS
            lbl_lines.append(f"{cls} {xc_canvas:.6f} {yc_canvas:.6f} {w_canvas:.6f} {h_canvas:.6f}")

        slot_idx += 1
        if slot_idx >= max_slots:
            flush_canvas()

    flush_canvas()
    return canvas_idx

# ---------------- pack: test (single-label whole image) ----------------
def pack_split_singlelabel(src_img_dir, src_lbl_dir, out_img_dir, out_lbl_dir):
    """
    รวมหลายภาพ (tile) ลงบนแคนวาส 640x640 เช่นกัน
    แต่ไฟล์ label เขียนเป็น 'กล่องเดียวเต็มภาพ' โดยใช้ class ของ tile แรกใน canvas
    """
    ensure_dir(out_img_dir)
    ensure_dir(out_lbl_dir)

    files = list_images(src_img_dir)
    if not files:
        return 0

    stride, cols, rows, max_slots = grid_params()

    canvas_idx = 0
    slot_idx = 0
    img_canvas = None
    cls_canvas = None  # คลาสเดียวสำหรับทั้งภาพ

    def flush_canvas():
        nonlocal img_canvas, canvas_idx, cls_canvas
        if img_canvas is None:
            return
        out_name = f"{canvas_idx:06d}.jpg"
        img_canvas.save(os.path.join(out_img_dir, out_name), "JPEG")
        # เขียน label เดียวเต็มภาพ
        with open(os.path.join(out_lbl_dir, out_name.replace(".jpg", ".txt")), "w", encoding="utf-8") as f:
            if cls_canvas is not None:
                f.write(f"{cls_canvas} 0.5 0.5 1 1\n")
            else:
                # ถ้าไม่มีคลาสเลย ให้เว้นว่าง (หรือจะไม่เขียนไฟล์ก็ได้ แล้วแต่ต้องการ)
                pass
        canvas_idx += 1
        img_canvas = None
        cls_canvas = None

    for fname in files:
        if img_canvas is None:
            img_canvas = Image.new("RGB", (CANVAS, CANVAS), (255, 255, 255))
            slot_idx = 0
            cls_canvas = None

        src_path = os.path.join(src_img_dir, fname)
        tile = Image.open(src_path).convert("RGB").resize((CELL, CELL))

        # อ่านคลาสของ tile ปัจจุบัน
        cls = read_first_class(os.path.join(src_lbl_dir, Path(fname).stem + ".txt"))
        if cls is None:
            # ไม่มี label ข้ามหรือจะกำหนด default class เองก็ได้
            pass
        else:
            if cls_canvas is None:
                cls_canvas = cls  # ใช้คลาสของ tile แรกใน canvas

        # ตำแหน่งลงบนกริด
        r = slot_idx // cols
        c = slot_idx % cols
        x = GAP + c * stride
        y = GAP + r * stride

        # ถ้าเกินขอบ แสดงว่าเต็มแล้ว → flush แล้วเริ่ม canvas ใหม่
        if x + CELL > CANVAS - GAP + 1 or y + CELL > CANVAS - GAP + 1:
            flush_canvas()
            img_canvas = Image.new("RGB", (CANVAS, CANVAS), (255, 255, 255))
            slot_idx = 0
            cls_canvas = None
            r = c = 0
            x = GAP
            y = GAP

        img_canvas.paste(tile, (x, y))
        slot_idx += 1

        if slot_idx >= max_slots:
            flush_canvas()

    flush_canvas()
    return canvas_idx

# ---------------- dataset.yaml helper ----------------
def extract_nc_names_from_yaml_text(y):
    """
    ดึง nc และ names จาก yaml ต้นฉบับแบบง่าย ๆ
    รองรับทั้ง 'name: [...]' และ 'names: [...]' (จะพยายามใช้ names ก่อน)
    """
    nc_line = None
    names_line = None

    lines = [ln.strip() for ln in y.splitlines() if ln.strip()]
    for ln in lines:
        if ln.startswith("nc:"):
            nc_line = ln
        # YOLO ปรกติใช้ 'names:' แต่บางที่เก่ามี 'name:'
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
    # force absolute paths
    train_abs = os.path.abspath(train_dir)
    val_abs   = os.path.abspath(val_dir)

    # default tail
    tail = "nc: 1\nnames: [class0]\n"

    if os.path.exists(src_yaml_path):
        with open(src_yaml_path, "r", encoding="utf-8") as f:
            y = f.read()
        # พยายามรักษา nc/names เดิม
        # 1) ลองดึงบล็อคตั้งแต่ 'nc:' จนจบ
        if "nc:" in y:
            tail = "nc:" + y.split("nc:", 1)[1]
            # กันเคส YAML เดิมใช้ 'name:' แทน 'names:' ของ YOLOv5
            tail = tail.replace("\nname:", "\nnames:")
        else:
            # ถ้าไม่มี nc: ใช้ tail เริ่มต้น
            pass

    new_yaml = f"train: {train_abs}\nval: {val_abs}\n{tail.strip()}\n"
    with open(out_yaml_path, "w", encoding="utf-8") as f:
        f.write(new_yaml)


# ---------------- main ----------------
def main():
    for src_root, dst_root in zip(SRC_KFOLDs, DST_KFOLDs):
        print(f"==> Processing {src_root} → {dst_root}")
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

            created_map = {}

            for sp in SPLITS:
                in_img = os.path.join(fold_src, sp, "images")
                in_lbl = os.path.join(fold_src, sp, "labels")

                out_img = os.path.join(fold_dst, sp, "images")
                out_lbl = os.path.join(fold_dst, sp, "labels")
                ensure_dir(out_img); ensure_dir(out_lbl)

                if sp == "test":
                    # ทำ test แบบ single-label ทั้งภาพ
                    created = pack_split_singlelabel(in_img, in_lbl, out_img, out_lbl)
                else:
                    # train/validation แบบหลายวัตถุ
                    created = pack_split(in_img, in_lbl, out_img, out_lbl)

                created_map[sp] = created
                print(f"    {sp}: {created} packed images")

            # dataset.yaml
            src_yaml = os.path.join(fold_src, "dataset.yaml")
            out_yaml = os.path.join(fold_dst, "dataset.yaml")
            write_dataset_yaml(
                src_yaml, out_yaml,
                os.path.join(dst_root, fold, "training"),
                os.path.join(dst_root, fold, "validation"),
            )

        print(f"✅ DONE {dst_root}\n")

if __name__ == "__main__":
    main()
