# clear.py
# Clear generated data on Windows / Linux / macOS (replaces rmdir /s /q and the docker rm commands)
#
#   python clear.py --host                        remove ./out and ./data next to this script, then recreate an empty ./out
#   python clear.py --volume                      remove the docker volume (images + split)
#   python clear.py --volume-images               keep split/, remove nopca/pca (images + manifests) in the volume
#   python clear.py --volume-datasets iris,wine   remove whole datasets (incl. split) from the volume
#   python clear.py --volume-k 8                  remove repacking_k_8 of every dataset/mode/algo in the volume
#   python clear.py --all                         --host + --volume
#   add --dry-run to only print, --yes to skip the confirmation

import argparse
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import config

NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_CFG = config.load()


def rmtree(path):
    def on_error(func, p, _exc):
        os.chmod(p, stat.S_IWRITE)  # Windows read-only files
        func(p)
    shutil.rmtree(path, onerror=on_error)


def docker(args, check=True):
    return subprocess.run(["docker", *args], check=check, text=True, capture_output=True)


def volume_exists(name):
    return docker(["volume", "inspect", name], check=False).returncode == 0


def in_volume(volume, shell_cmd):
    docker(["run", "--rm", "-v", f"{volume}:/out", "alpine", "sh", "-c", shell_cmd])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", action="store_true", help="remove out/ and data/ folders")
    ap.add_argument("--volume", action="store_true", help="remove the whole docker volume")
    ap.add_argument("--volume-images", action="store_true", help="remove nopca/pca of every dataset, keep split/")
    ap.add_argument("--volume-datasets", default=None, help="comma-separated datasets to remove from the volume")
    ap.add_argument("--volume-k", type=int, default=None, help="remove repacking_k_<k> everywhere in the volume")
    ap.add_argument("--all", action="store_true", help="--host and --volume")
    ap.add_argument("--root", default=None, help="folder holding the export and data folders (default: paths in config.json)")
    ap.add_argument("--volume-name", default=_CFG["volume"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", "-y", action="store_true", help="do not ask for confirmation")
    args = ap.parse_args()

    if args.all:
        args.host = args.volume = True

    vol = args.volume_name
    steps = []  # (description, callable)

    if args.host:
        names = [_CFG["export_dir"], _CFG["data_dir"]]
        folders = [Path(args.root) / Path(n).name if args.root else config.resolve(n) for n in names]
        for p in folders:
            steps.append((f"delete folder {p}", (lambda p=p: rmtree(p) if p.exists() else print(f"  {p} not found"))))
        out_dir = folders[0]  # export_dir: recreate it empty, owned by the current user (docker would make it root's)
        steps.append((f"recreate empty folder {out_dir}", lambda: out_dir.mkdir(parents=True, exist_ok=True)))

    docker_steps = []
    if args.volume:
        docker_steps.append((f"delete docker volume {vol}",
                             lambda: docker(["volume", "rm", vol]) if volume_exists(vol) else print(f"  volume {vol} not found")))
    else:
        if args.volume_images:
            docker_steps.append((f"volume {vol}: remove nopca/pca of all datasets (keep split)",
                                 lambda: in_volume(vol, "rm -rf /out/*/nopca /out/*/pca")))
        if args.volume_datasets:
            names = [d.strip() for d in args.volume_datasets.split(",") if d.strip()]
            bad = [n for n in names if not NAME_RE.match(n)]
            if bad:
                sys.exit(f"invalid dataset name: {bad}")
            for n in names:
                docker_steps.append((f"volume {vol}: remove dataset {n} (incl. split)",
                                     lambda n=n: in_volume(vol, f"rm -rf /out/{n}")))
        if args.volume_k is not None:
            docker_steps.append((f"volume {vol}: remove repacking_k_{args.volume_k} everywhere",
                                 lambda: in_volume(vol, f"rm -rf /out/*/*/algo*/repacking_k_{args.volume_k}")))
    steps += docker_steps

    if not steps:
        ap.print_help()
        sys.exit(1)

    print("will do:")
    for desc, _ in steps:
        print("  -", desc)
    if args.dry_run:
        print("(dry run, nothing deleted)")
        return
    if not args.yes and input("continue? [y/N] ").strip().lower() != "y":
        print("cancelled")
        return

    if docker_steps and shutil.which("docker") is None:
        sys.exit("docker not found in PATH")
    for desc, fn in steps:
        print("*", desc)
        try:
            fn()
        except subprocess.CalledProcessError as e:
            sys.exit(f"failed: {(e.stderr or '').strip() or e}")
    print("done")


if __name__ == "__main__":
    main()
