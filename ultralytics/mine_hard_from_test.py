# mine_hard_from_test.py
from pathlib import Path
import argparse
import re

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def latest_val_labels_dir(runs_detect_dir: Path) -> Path:
    # กรณีส่งมาที่โฟลเดอร์ val_xxx โดยตรง
    direct = runs_detect_dir / "labels"
    if direct.exists():
        return direct

    if not runs_detect_dir.exists():
        raise SystemExit(f"runs/detect not found: {runs_detect_dir}")

    candidates = []
    for p in runs_detect_dir.iterdir():
        # ยอมรับทุกโฟลเดอร์ที่ขึ้นต้นด้วย val (val, val2, val_test2_round1, ...)
        if p.is_dir() and re.match(r"^val.*$", p.name, re.IGNORECASE):
            labels = p / "labels"
            if labels.exists():
                candidates.append(labels)

    if not candidates:
        raise SystemExit(f"No val*/labels folder under {runs_detect_dir}")

    candidates.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return candidates[0]


def img_to_rel_from_images(img_path: Path) -> Path:
    """
    คืนค่า path ส่วนที่อยู่ 'หลังคำว่า images/' เพื่อใช้สร้างทางไปยัง labels
    เช่น:
      .../images/a/b/c.jpg -> a/b/c.jpg
    ถ้าไม่มีคำว่า images ให้คืนเฉพาะชื่อไฟล์ (c.jpg)
    """
    parts_lower = [p.lower() for p in img_path.parts]
    if "images" in parts_lower:
        i = parts_lower.index("images")
        rel = Path(*img_path.parts[i+1:])
        return rel
    return Path(img_path.name)

def main():
    ap = argparse.ArgumentParser(description="Select hard cases from test2 predictions into a txt list.")
    ap.add_argument("--test2_list", required=True, help="path to _splits2\\test2.txt")
    ap.add_argument("--gt_labels_root", required=True, help="root of GT labels (e.g. ...\\training\\labels)")
    ap.add_argument("--runs_detect_dir", default=r"C:\yolo8_strip\ultralytics\runs\detect",
                    help="Ultralytics runs/detect directory")
    ap.add_argument("--out_hard_txt", required=True, help="output txt path for hard image list")
    ap.add_argument("--conf_threshold", type=float, default=0.25, help="mark hard if max conf < this")
    args = ap.parse_args()

    test2_list   = Path(args.test2_list)
    labels_root  = Path(args.gt_labels_root)
    runs_detect  = Path(args.runs_detect_dir)
    out_hard_txt = Path(args.out_hard_txt)
    thr          = args.conf_threshold

    if not test2_list.exists():
        raise SystemExit(f"test2_list not found: {test2_list}")
    if not labels_root.exists():
        raise SystemExit(f"GT labels root not found: {labels_root}")

    # หาโฟลเดอร์ผลทำนายล่าสุดอัตโนมัติ เช่น runs/detect/val3/labels
    pred_root = latest_val_labels_dir(runs_detect)

    # โหลดรายการภาพ test2
    test_images = [Path(s.strip()) for s in test2_list.read_text(encoding="utf-8").splitlines() if s.strip()]

    hard = []
    missing_pred = 0
    missing_gt = 0

    for img in test_images:
        if img.suffix.lower() not in IMG_EXTS:
            continue

        # หา path ของ label GT: เอาส่วน rel หลัง images/ มาแปะใต้ labels_root แล้วเปลี่ยน .ext -> .txt
        rel_from_images = img_to_rel_from_images(img)
        gt = (labels_root / rel_from_images).with_suffix(".txt")

        # ไฟล์ทำนายของ YOLOv8 ถูกเซฟที่ pred_root/<image_stem>.txt (มักไม่เก็บโครง subfolder)
        pred = pred_root / (img.stem + ".txt")

        # ต้องมี GT และมีบรรทัด
        if not gt.exists() or gt.read_text(encoding="utf-8").strip() == "":
            missing_gt += 1
            continue  # ไม่มี GT ก็ไม่ถือว่าเป็น hard

        # ถ้าไม่มี prediction เลย → hard
        if not pred.exists() or pred.read_text(encoding="utf-8").strip() == "":
            hard.append(str(img))
            missing_pred += 1
            continue

        # หา max conf จากไฟล์ทำนาย: class x y w h conf [optional: ...]
        max_conf = 0.0
        for line in pred.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) >= 6:
                try:
                    conf = float(parts[5])
                    if conf > max_conf:
                        max_conf = conf
                except ValueError:
                    pass

        if max_conf < thr:
            hard.append(str(img))

    # กันซ้ำ/เรียง
    hard_unique_sorted = sorted(set(hard))

    out_hard_txt.parent.mkdir(parents=True, exist_ok=True)
    out_hard_txt.write_text("\n".join(hard_unique_sorted), encoding="utf-8")

    print(f"[mine_hard_from_test] pred_root = {pred_root}")
    print(f"Test images     : {len(test_images)}")
    print(f"Hard (no pred)  : {missing_pred}")
    print(f"Missing GT skip : {missing_gt}")
    print(f"Hard total      : {len(hard_unique_sorted)} -> {out_hard_txt}")

if __name__ == "__main__":
    main()
