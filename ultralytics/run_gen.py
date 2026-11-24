# run_gen.py
# One command for split / gen via docker (Windows + Linux). Replaces the long `docker compose run ...` lines.
#
#   python run_gen.py all                          init data + build image (if missing) + split + gen, every dataset found in data/
#   python run_gen.py split --seed 42              split only
#   python run_gen.py gen --k_list 0,2,4,8         gen only (needs split in the volume)
#   python run_gen.py build                        (re)build the image
#
# common options
#   --datasets a,b,c   default: every known dataset whose csv is in ./data
#   --modes nopca,pca  --algos 1,2  --k_list 0,2,4,8  --seed 42  --cell auto  --no_jpeg_roundtrip
#   --preview          gen: also write <fold>/preview/ (images with boxes + row_ids drawn on; bigger, not used by YOLO)
#   --overwrite        replace an existing split
#   --volume stripout  --image strip-gen  --data data
#   --dry-run          print the docker commands only

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import config
import init_data

HERE = Path(__file__).resolve().parent
_CFG = config.load()
_DEF = _CFG["defaults"]


def docker(args, check=True, quiet=False):
    r = subprocess.run(["docker", *args], text=True, capture_output=quiet)
    if check and r.returncode != 0:
        sys.exit(f"docker failed (exit {r.returncode})" + (f": {r.stderr.strip()}" if quiet else ""))
    return r


def image_exists(image):
    return docker(["image", "inspect", image], check=False, quiet=True).returncode == 0


def found_datasets(data_dir):
    return [d for d, csv in init_data.CSV_NAMES.items() if (data_dir / d / csv).is_file()]


def configured_datasets():
    """config.json "run_datasets": the default list when --datasets is not given (optional key)"""
    names = _CFG.get("run_datasets") or []
    unknown = [n for n in names if n not in init_data.CSV_NAMES]
    if unknown:
        sys.exit(f"config.json run_datasets has names missing from datasets: {', '.join(unknown)}")
    return list(names)


def pick_datasets(a, data_dir):
    """priority: --datasets  >  config.json run_datasets  >  datasets whose csv is in data/"""
    if a.datasets:
        return [d.strip() for d in a.datasets.split(",") if d.strip()]
    names = configured_datasets()
    if names:
        return names
    names = found_datasets(data_dir)
    if not names:
        sys.exit(f"no dataset csv in {data_dir} (run: python init_data.py --all, or set run_datasets in config.json)")
    return names


def run_tool(a, data_dir, tool_args, mount_data):
    cmd = ["run", "--rm", "--mount", f"type=volume,src={a.volume},dst=/out"]
    if mount_data:
        cmd += ["--mount", f"type=bind,src={data_dir},dst=/data"]
    cmd += [a.image, *tool_args]
    if a.dry_run:
        print("docker", " ".join(cmd))
        return
    docker(cmd)


def do_build(a):
    cmd = ["build", "-t", a.image, str(HERE)]
    if a.dry_run:
        print("docker", " ".join(cmd))
        return
    docker(cmd)


def do_split(a, data_dir, names):
    args = ["split", "--datasets", ",".join(names), "--dataset_root", "/data", "--out_dir", "/out",
            "--seed", str(a.seed)]
    if a.overwrite:
        args.append("--overwrite")
    run_tool(a, data_dir, args, mount_data=True)


def do_gen(a, data_dir, names):
    args = ["gen", "--datasets", ",".join(names), "--out_dir", "/out", "--modes", a.modes, "--algos", a.algos,
            "--k_list", a.k_list, "--cell", a.cell, "--seed", str(a.gen_seed)]
    if a.no_jpeg_roundtrip:
        args.append("--no_jpeg_roundtrip")
    if a.preview:
        args.append("--preview")
    run_tool(a, data_dir, args, mount_data=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["all", "split", "gen", "build"])
    ap.add_argument("--datasets", default=None)
    ap.add_argument("--modes", default=_DEF["modes"])
    ap.add_argument("--algos", default=_DEF["algos"])
    ap.add_argument("--k_list", default=_DEF["k_list"])
    ap.add_argument("--seed", type=int, default=_DEF["split_seed"], help="split seed")
    ap.add_argument("--gen-seed", dest="gen_seed", type=int, default=_DEF["gen_seed"], help="seed of the repack shuffles in gen")
    ap.add_argument("--cell", default=_DEF["cell"])
    ap.add_argument("--no_jpeg_roundtrip", action="store_true")
    ap.add_argument("--preview", action="store_true", help="gen: also write preview/ images with boxes + row_ids drawn on")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--volume", default=_CFG["volume"])
    ap.add_argument("--image", default=_CFG["image"])
    ap.add_argument("--data", default=str(config.resolve(_CFG["data_dir"])))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if shutil.which("docker") is None:
        sys.exit("docker not found in PATH")
    data_dir = Path(a.data).resolve()

    if not a.dry_run:
        # create the export folder as the current user, so docker never creates it as root
        config.resolve(_CFG["export_dir"]).mkdir(parents=True, exist_ok=True)

    if a.step == "build":
        do_build(a)
        return

    if a.step == "all":
        if not a.dry_run:
            init_data_args = [sys.executable, str(HERE / "init_data.py"), "--dest", str(data_dir)]
            init_data_args += ["--datasets", ",".join(pick_datasets(a, data_dir))] if (a.datasets or configured_datasets()) else ["--all"]
            subprocess.run(init_data_args, check=True)
        if a.dry_run or not image_exists(a.image):
            do_build(a)

    if a.step in ("all", "split"):
        names = pick_datasets(a, data_dir)
        print(f"== split: {','.join(names)}")
        do_split(a, data_dir, names)

    if a.step in ("all", "gen"):
        names = pick_datasets(a, data_dir)
        print(f"== gen: {','.join(names)}")
        do_gen(a, data_dir, names)

    print("done. results are in docker volume:", a.volume)


if __name__ == "__main__":
    main()
