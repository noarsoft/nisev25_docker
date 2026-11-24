# init_data.py
# Prepare <dest>/<dataset>/<csv> for gen_folds.py / docker (replaces: mkdir data\iris + copy iris.csv data\iris\)
#   python init_data.py --datasets iris
#   python init_data.py --all                 (every dataset whose csv is present)
#   python init_data.py --datasets iris,wine,ionosphere --src . --dest data

import argparse
import shutil
import sys
from pathlib import Path

import config

_CFG = config.load()
# dataset name -> raw csv file name (config.json "datasets")
CSV_NAMES = _CFG["datasets"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default=None, help="comma-separated, e.g. iris,wine")
    ap.add_argument("--all", action="store_true", help="every known dataset whose csv exists in --src; the others are skipped")
    ap.add_argument("--src", default=str(Path(__file__).resolve().parent), help="folder that holds the csv files")
    ap.add_argument("--dest", default=str(config.resolve(_CFG["data_dir"])), help="dataset_root to create (default: data_dir in config.json)")
    ap.add_argument("--file", default=None, help="csv file name to use instead of the default (single dataset only)")
    args = ap.parse_args()

    if bool(args.datasets) == args.all:
        sys.exit("give either --datasets or --all")
    if args.all:
        names = [d for d in CSV_NAMES if (Path(args.src) / CSV_NAMES[d]).is_file()]
        print("datasets found:", ",".join(names))
    else:
        names = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if args.file and len(names) != 1:
        sys.exit("--file can only be used with a single dataset")

    failed = False
    for ds in names:
        if ds not in CSV_NAMES:
            print(f"[{ds}] unknown dataset. Known: {', '.join(CSV_NAMES)}")
            failed = True
            continue
        csv_name = CSV_NAMES[ds]
        src = Path(args.src) / (args.file or csv_name)
        if not src.is_file():
            print(f"[{ds}] not found: {src}")
            failed = True
            continue
        dst_dir = Path(args.dest) / ds
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / csv_name)
        print(f"[{ds}] {src} -> {dst_dir / csv_name}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
