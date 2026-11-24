# make_train_with_hard.py
from pathlib import Path
import argparse
import random
import yaml

def read_nonempty_lines(p):
    if not p.exists():
        raise SystemExit(f"File not found: {p}")
    return [s.strip() for s in p.read_text(encoding="utf-8").splitlines() if s.strip()]

def main():
    ap = argparse.ArgumentParser(description="Combine train2 + hard(test2) with oversample; keep val2 for monitor.")
    ap.add_argument("--splits_dir", required=True,
                    help="_splits2 fold ex. C:\\...\\output_k_fold_boost\\0\\_splits2")
    ap.add_argument("--oversample_k", type=int, default=1,
                    help="repeate training oversample (>=1)")
    ap.add_argument("--seed", type=int, default=42,
                    help="seed train")
    args = ap.parse_args()

    splits_dir = Path(args.splits_dir)
    train2_list = splits_dir / "train2.txt"
    val2_list   = splits_dir / "val2.txt"
    test2_list  = splits_dir / "test2.txt"
    hard_list   = splits_dir / "hard_from_test2.txt"
    ds2_yaml    = splits_dir / "dataset2.yaml"

    # อ่านรายการ
    train2 = read_nonempty_lines(train2_list)
    val2   = set(read_nonempty_lines(val2_list))     # set เพื่อกันรั่ว
    test2  = set(read_nonempty_lines(test2_list))
    hard   = read_nonempty_lines(hard_list)

    # กันรั่ว: hard ที่ชนกับ val2 จะถูกตัดทิ้ง (เผื่ออนาคตมีการคัด hard จาก val บ้าง)
    hard_filtered = [p for p in hard if p not in val2]

    # oversample แบบควบคุม (ซ้ำ K รอบ)
    k = max(1, args.oversample_k)
    train_next = list(train2) + hard_filtered * k

    # สุ่มลำดับเพื่อป้องกันลำเอียง
    random.seed(args.seed)
    random.shuffle(train_next)

    # เขียนไฟล์ผลลัพธ์
    out_train = splits_dir / "train_with_hard.txt"
    out_val   = splits_dir / "val2.txt"  # ใช้ val2 เดิม
    out_train.write_text("\n".join(train_next), encoding="utf-8")

    # โหลด dataset2.yaml เพื่อดึง nc/names/test
    if not ds2_yaml.exists():
        raise SystemExit(f"dataset2.yaml not found: {ds2_yaml}")
    ds2 = yaml.safe_load(ds2_yaml.read_text(encoding="utf-8"))

    # สร้าง dataset_with_hard.yaml
    ds_hard = {
        "train": str(out_train),
        "val":   str(out_val),
        "test":  ds2.get("test", str(test2_list)),
        "nc":    ds2.get("nc"),
        "names": ds2.get("names") or ds2.get("name"),
    }
    out_yaml = splits_dir / "dataset_with_hard.yaml"
    out_yaml.write_text(yaml.safe_dump(ds_hard, sort_keys=False, allow_unicode=True), encoding="utf-8")

    # สรุปผล
    print("=== make_train_with_hard ===")
    print(f"splits_dir       : {splits_dir}")
    print(f"train2           : {len(train2)}")
    print(f"val2             : {len(val2)}")
    print(f"test2            : {len(test2)}")
    print(f"hard (input)     : {len(hard)}")
    print(f"hard (filtered)  : {len(hard_filtered)}  (removed {len(hard)-len(hard_filtered)} overlapped with val2)")
    print(f"oversample_k     : {k}")
    print(f"train_with_hard  : {len(train_next)} -> {out_train}")
    print(f"val (monitor)    : {len(val2)} -> {out_val}")
    print(f"dataset_with_hard: {out_yaml}")

if __name__ == "__main__":
    main()
