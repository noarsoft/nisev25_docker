
import argparse
import csv
import re
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

# python count_repacking_images.py --datasets breast_cancer,iris,wine,automobile,ionosphere

DEFAULT_DATASETS = [
    "breast_cancer",
    "iris",
    "wine",
    "automobile",
    "ionosphere",
]

ALGORITHMS = [
    ("output_k_fold_boost_nopca", "No-PCA (Algorithm 1)"),
    ("output_k_fold_boost_pca", "PCA (Algorithm 1)"),
    ("output_k_fold_boost_nopca2", "No-PCA (Algorithm 2)"),
    ("output_k_fold_boost_pca2", "PCA (Algorithm 2)"),
]

SPLITS = ("training", "validation", "test")

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tif", ".tiff",
}

K_PATTERN = re.compile(r"repacking_k_(\d+)$")


# ============================================================
# COUNT IMAGES
# ============================================================

def count_images(fold_dir, split):
    split_dir = fold_dir / split

    if not split_dir.is_dir():
        return 0, False

    # รองรับทั้ง training/images และ training โดยตรง
    image_dir = split_dir / "images"

    if not image_dir.is_dir():
        image_dir = split_dir

    total = sum(
        1
        for path in image_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    return total, True


def find_k_dirs(dataset_dir, k_filter=None):
    results = []

    if not dataset_dir.is_dir():
        return results

    for folder in dataset_dir.iterdir():
        if not folder.is_dir():
            continue

        found = K_PATTERN.fullmatch(folder.name)

        if not found:
            continue

        k = int(found.group(1))

        if k_filter is None or k in k_filter:
            results.append((k, folder))

    return sorted(results, key=lambda x: x[0])


# ============================================================
# TERMINAL TABLE
# ============================================================

def print_table(headers, rows):
    rows = [[str(v) for v in row] for row in rows]

    widths = [
        max(
            [len(str(headers[i]))]
            + [len(row[i]) for row in rows]
        )
        for i in range(len(headers))
    ]

    def format_row(row):
        return "  ".join(
            str(value).ljust(widths[i])
            for i, value in enumerate(row)
        )

    print(format_row(headers))
    print("  ".join("-" * width for width in widths))

    for row in rows:
        print(format_row(row))


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(path, records, columns):
    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=columns
        )

        writer.writeheader()
        writer.writerows(records)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent
    )

    parser.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="Dataset names separated by comma, or all"
    )

    parser.add_argument(
        "--k_list",
        default="all",
        help="Example: 0,2,4,8,16 or all"
    )

    args = parser.parse_args()

    root = args.root.resolve()

    if not root.is_dir():
        parser.error(f"Root folder not found: {root}")

    output_dir = root / "image_count_reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset selection
    if args.datasets.lower() == "all":
        datasets = sorted(
            folder.name
            for folder in root.iterdir()
            if folder.is_dir()
            and find_k_dirs(folder)
        )
    else:
        datasets = [
            name.strip()
            for name in args.datasets.split(",")
            if name.strip()
        ]

    # K selection
    if args.k_list.lower() == "all":
        k_filter = None
    else:
        k_filter = {
            int(k.strip())
            for k in args.k_list.split(",")
            if k.strip()
        }

    per_fold_records = []
    summary_records = []
    warnings = []

    print("\n" + "=" * 90)
    print("NISE / NISEv2 IMAGE COUNTER")
    print("=" * 90)
    print(f"Root: {root}")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"K values: {args.k_list}")

    # Loop datasets
    for dataset in datasets:

        dataset_dir = root / dataset
        k_dirs = find_k_dirs(dataset_dir, k_filter)

        if not dataset_dir.exists():
            warnings.append(
                f"Dataset missing: {dataset_dir}"
            )
            continue

        if not k_dirs:
            warnings.append(
                f"No repacking folders: {dataset_dir}"
            )
            continue

        # Loop repacking k
        for k, k_dir in k_dirs:

            # Loop algorithms
            for algo_folder, algo_name in ALGORITHMS:

                algo_dir = k_dir / algo_folder

                if algo_dir.is_dir():
                    fold_dirs = sorted(
                        [
                            p for p in algo_dir.iterdir()
                            if p.is_dir() and p.name.isdigit()
                        ],
                        key=lambda p: int(p.name)
                    )
                else:
                    fold_dirs = []

                totals = {
                    "training": 0,
                    "validation": 0,
                    "test": 0,
                }

                fold_rows = []

                if not fold_dirs:
                    warnings.append(
                        f"Missing algorithm/folds: {algo_dir}"
                    )

                # Loop folds
                for fold_dir in fold_dirs:

                    counts = {}

                    for split in SPLITS:

                        count, exists = count_images(
                            fold_dir,
                            split
                        )

                        counts[split] = count
                        totals[split] += count

                        if not exists:
                            warnings.append(
                                f"Missing split: {fold_dir / split}"
                            )

                    total_fold = sum(counts.values())

                    # Store fold result
                    per_fold_records.append({
                        "dataset": dataset,
                        "k": k,
                        "algorithm": algo_name,
                        "algorithm_folder": algo_folder,
                        "fold": fold_dir.name,
                        "training": counts["training"],
                        "validation": counts["validation"],
                        "test": counts["test"],
                        "total": total_fold,
                    })

                    fold_rows.append([
                        fold_dir.name,
                        counts["training"],
                        counts["validation"],
                        counts["test"],
                        total_fold,
                    ])

                total_all = sum(totals.values())

                # Store summary
                summary_records.append({
                    "dataset": dataset,
                    "k": k,
                    "algorithm": algo_name,
                    "algorithm_folder": algo_folder,
                    "fold_count": len(fold_dirs),
                    "training": totals["training"],
                    "validation": totals["validation"],
                    "test": totals["test"],
                    "total": total_all,
                    "status": (
                        "OK" if fold_dirs
                        else "MISSING_OR_EMPTY"
                    ),
                })

                # Print individual table
                print("\n" + "=" * 90)
                print(
                    f"DATASET: {dataset} | "
                    f"K = {k} | {algo_name}"
                )
                print("=" * 90)

                fold_rows.append([
                    "TOTAL",
                    totals["training"],
                    totals["validation"],
                    totals["test"],
                    total_all,
                ])

                print_table(
                    [
                        "Fold",
                        "Training",
                        "Validation",
                        "Test",
                        "Total",
                    ],
                    fold_rows
                )

                if not fold_dirs:
                    print("[WARNING] No fold folders found")

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    save_csv(
        output_dir / "image_count_per_fold.csv",
        per_fold_records,
        [
            "dataset",
            "k",
            "algorithm",
            "algorithm_folder",
            "fold",
            "training",
            "validation",
            "test",
            "total",
        ]
    )

    save_csv(
        output_dir / "image_count_summary.csv",
        summary_records,
        [
            "dataset",
            "k",
            "algorithm",
            "algorithm_folder",
            "fold_count",
            "training",
            "validation",
            "test",
            "total",
            "status",
        ]
    )

    warnings_path = output_dir / "image_count_warnings.txt"

    warnings_path.write_text(
        "\n".join(warnings) if warnings else "No warnings.",
        encoding="utf-8"
    )

    # ========================================================
    # PRINT FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 110)
    print("FINAL SUMMARY - ALL DATASETS / ALL K / ALL ALGORITHMS")
    print("=" * 110)

    rows = []

    for record in summary_records:

        rows.append([
            record["dataset"],
            record["k"],
            record["algorithm"],
            record["fold_count"],
            record["training"],
            record["validation"],
            record["test"],
            record["total"],
        ])

    print_table(
        [
            "Dataset",
            "K",
            "Algorithm",
            "Folds",
            "Train",
            "Val",
            "Test",
            "Total",
        ],
        rows
    )

    print("\nRESULT FILES:")
    print(output_dir / "image_count_per_fold.csv")
    print(output_dir / "image_count_summary.csv")
    print(output_dir / "image_count_warnings.txt")

    print(f"\nWarnings: {len(warnings)}")
    print("DONE!")


if __name__ == "__main__":
    main()
