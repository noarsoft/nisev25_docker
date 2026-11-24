# transfer.py
# Export / import the docker volume (default: stripout). Same script on Windows and Linux.
#
#   python transfer.py export                          split only (CSV + manifest, small)  -> out/split_backup.tar.gz
#   python transfer.py export --what all               everything incl. images             -> out/strip_all.tar.gz
#   python transfer.py list                            list repacking_k_* folders in the volume
#   python transfer.py export --what folder --path iris/nopca/algo1/repacking_k_8
#                                                      copy folder(s) as real folders (no tar file); several paths and * allowed
#   python transfer.py export --what folder --path "iris/*/algo*/repacking_k_8" "wine/pca/algo1/repacking_k_8"
#   python transfer.py export --what sets --datasets iris,wine --modes nopca --algos 1,2 --k_list 8
#                                                      every dataset/mode/algo/k combination that exists (omitted options use config.json defaults)
#                                                      the algo manifest.json (Ls/Hs per fold) is copied along
#   python transfer.py export --what extras --datasets iris --algos 1,2 --k_list 8 [--parts index|preview|index,preview]
#                                                      only <fold>/index and <fold>/preview of the sets (from gen --preview); the
#                                                      training/validation/test folders already in out/ are not touched
#   add --move to folder/sets: after the copy is verified (file count + total size per folder) the same
#   repacking_k_* folders are deleted from the volume; split/ and manifest.json stay, so gen can recreate them
#   python transfer.py export --what data              ./data on this machine (no docker)  -> out/data_backup.tar.gz
#   python transfer.py import out/split_backup.tar.gz  extract into the docker volume
#   python transfer.py import out/split_backup.tar.gz --to-dir out   extract on the host instead (no docker)
#
# options: --dest DIR (default ./out)  --volume NAME (default stripout)  --name FILE

import argparse
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path

import config

IMAGE = "alpine"
PATH_RE = re.compile(r"^[A-Za-z0-9_./*?-]+$")
HERE = Path(__file__).resolve().parent
_CFG = config.load()


def docker(args, check=True, quiet=False):
    r = subprocess.run(["docker", *args], text=True, capture_output=quiet)
    if check and r.returncode != 0:
        sys.exit(f"docker failed (exit {r.returncode})" + (f": {r.stderr.strip()}" if quiet else ""))
    return r


def volume_exists(name):
    return docker(["volume", "inspect", name], check=False, quiet=True).returncode == 0


def run_in_container(volume, host_dir, shell_cmd, volume_ro, host_ro, as_user):
    cmd = ["run", "--rm"]
    if as_user and hasattr(os, "getuid"):
        cmd += ["--user", f"{os.getuid()}:{os.getgid()}"]  # keep exported files owned by you on Linux
    vol = f"type=volume,src={volume},dst=/out" + (",readonly" if volume_ro else "")
    bind = f"type=bind,src={host_dir},dst=/export" + (",readonly" if host_ro else "")
    cmd += ["--mount", vol, "--mount", bind, IMAGE, "sh", "-c", shell_cmd]
    docker(cmd)


def safe_extract(tar, dest):
    dest = dest.resolve()
    for m in tar.getmembers():
        target = (dest / m.name).resolve()
        if dest != target and dest not in target.parents:
            sys.exit(f"unsafe path in archive: {m.name}")
    tar.extractall(dest)


def list_in_volume(volume, patterns):
    """expand shell patterns (relative to the volume root) inside a container; returns existing paths"""
    out = []
    for pat in patterns:
        p = pat.strip().strip("/")
        if not p or ".." in p.split("/") or not PATH_RE.match(p):
            sys.exit(f"bad path/pattern: {pat!r}")
        cmd = ["run", "--rm", "--mount", f"type=volume,src={volume},dst=/out,readonly", IMAGE,
               "sh", "-c", f"cd /out && ls -d {p} 2>/dev/null"]
        r = docker(cmd, check=False, quiet=True)
        found = [x for x in r.stdout.split("\n") if x.strip()]
        if not found:
            print(f"  not found: {p}")
        out += found
    return sorted(dict.fromkeys(out))


def stream_to_host(volume, paths, dest):
    """docker streams a tar to stdout and python unpacks it straight into dest (no archive file, no bind mount)"""
    dest = dest.resolve()
    cmd = ["docker", "run", "--rm", "--mount", f"type=volume,src={volume},dst=/out,readonly", IMAGE,
           "tar", "cf", "-", "-C", "/out", *paths]
    n_files = 0
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as t:
            for m in t:
                target = (dest / m.name).resolve()
                if dest != target and dest not in target.parents:
                    proc.kill()
                    sys.exit(f"unsafe path in stream: {m.name}")
                if m.isdir() or m.isfile():
                    t.extract(m, dest)
                    n_files += m.isfile()
        err = proc.stderr.read().decode(errors="replace").strip()
        if proc.wait() != 0:
            sys.exit(f"docker tar failed: {err}")
    return n_files


def algo_manifests(volume, folders):
    """manifest.json of the algo folder above each repacking_k_* folder (Ls/Hs per fold, needed for predict)"""
    found = list_in_volume(volume, ["*/*/algo*/manifest.json"])
    wanted = {str(Path(f).parent).replace("\\", "/") + "/manifest.json" for f in folders}
    return [m for m in found if m in wanted]


def volume_stats(volume, folders):
    """folder -> (file count, total bytes) as stored in the volume"""
    shell = "cd /out && for p in " + " ".join(folders) + "; do " \
            "n=$(find $p -type f | wc -l); " \
            "s=$(find $p -type f -exec stat -c %s {} + | awk '{t+=$1} END{print t+0}'); " \
            "echo \"$p $n $s\"; done"
    r = docker(["run", "--rm", "--mount", f"type=volume,src={volume},dst=/out,readonly", IMAGE, "sh", "-c", shell],
               quiet=True)
    stats = {}
    for line in r.stdout.splitlines():
        p, n, s = line.rsplit(" ", 2)
        stats[p] = (int(n), int(s))
    return stats


def host_stats(dest, folder):
    files = [f for f in (dest / folder).rglob("*") if f.is_file()]
    return len(files), sum(f.stat().st_size for f in files)


def verified_folders(volume, folders, dest):
    """folders whose copy on the host has the same file count and total size as the volume"""
    vol = volume_stats(volume, folders)
    ok = []
    for f in folders:
        h = host_stats(dest, f)
        if vol.get(f) == h and h[0] > 0:
            ok.append(f)
        else:
            print(f"  MISMATCH {f}: volume={vol.get(f)} host={h} -> kept in the volume")
    return ok


def delete_in_volume(volume, folders):
    for f in folders:
        if not re.match(r"^[^/]+/[^/]+/algo[^/]+/repacking_k_[0-9]+$", f):  # only ever delete image folders
            sys.exit(f"refusing to delete {f}")
        docker(["run", "--rm", "--mount", f"type=volume,src={volume},dst=/out", IMAGE, "sh", "-c",
                f"rm -rf /out/{f}"], quiet=True)


def cmd_list(a):
    if not volume_exists(a.volume):
        sys.exit(f"volume {a.volume} not found")
    for p in list_in_volume(a.volume, ["*/*/algo*/repacking_k_*"]):
        print(p)


def cmd_export(a):
    dest = Path(a.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)  # create as the current user, not as root via docker

    if a.what == "data":
        name = a.name or "data_backup.tar.gz"
        src = config.resolve(_CFG["data_dir"])
        if not src.is_dir():
            sys.exit(f"{src} not found")
        with tarfile.open(dest / name, "w:gz") as t:
            t.add(src, arcname="data")
        print(f"wrote {dest / name}")
        return

    if not volume_exists(a.volume):
        sys.exit(f"volume {a.volume} not found")

    if a.what == "extras":
        if a.move:
            sys.exit("--move is not supported for extras (index/preview are small; nothing is deleted)")
        parts = [p.strip() for p in a.parts.split(",") if p.strip()]
        if not parts or any(p not in ("index", "preview") for p in parts):
            sys.exit("--parts must be index, preview or index,preview")
        ds = a.datasets or "*"
        patterns = [f"{d}/{m}/algo{al}/repacking_k_{k}/*/{p}"
                    for d in ds.split(",") for m in a.modes.split(",")
                    for al in a.algos.split(",") for k in a.k_list.split(",") for p in parts]
        folders = list_in_volume(a.volume, patterns)
        if not folders:
            sys.exit("nothing to export (gen with --preview first; index/ is written by every gen)")
        print(f"exporting {len(folders)} folder(s) to {dest} ...")
        print(f"done: {stream_to_host(a.volume, folders, dest)} files (training/validation/test folders are not touched)")
        return

    if a.what in ("folder", "sets"):
        if a.what == "folder":
            if not a.path:
                sys.exit("--path is required, e.g. --path iris/nopca/algo1/repacking_k_8 (several and * allowed)")
            patterns = a.path
        else:
            ds = a.datasets or "*"
            patterns = [f"{d}/{m}/algo{al}/repacking_k_{k}"
                        for d in ds.split(",") for m in a.modes.split(",")
                        for al in a.algos.split(",") for k in a.k_list.split(",")]
        folders = list_in_volume(a.volume, patterns)
        if not folders:
            sys.exit("nothing to export (see: python transfer.py list)")
        paths = folders + algo_manifests(a.volume, folders)
        print(f"exporting {len(folders)} folder(s) to {dest} ...")
        n = stream_to_host(a.volume, paths, dest)
        print(f"done: {n} files")
        if a.move:
            print("verifying the copy before deleting from the volume ...")
            ok = verified_folders(a.volume, folders, dest)
            if ok:
                delete_in_volume(a.volume, ok)
            print(f"moved {len(ok)}/{len(folders)} folder(s); split/ and algo manifest.json stay in the volume")
            if len(ok) != len(folders):
                sys.exit(1)
        return

    if a.what == "split":
        name = a.name or "split_backup.tar.gz"
        shell = f"cd /out && tar czf /export/{name} */split"
    else:
        name = a.name or "strip_all.tar.gz"
        shell = f"tar czf /export/{name} -C /out ."

    run_in_container(a.volume, dest, shell, volume_ro=True, host_ro=False, as_user=True)
    print(f"exported to {dest / name}")


def cmd_import(a):
    f = Path(a.file).resolve()
    if not f.is_file():
        sys.exit(f"{f} not found")

    if a.to_dir:
        dest = Path(a.to_dir).resolve()
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(f, "r:gz") as t:
            safe_extract(t, dest)
        print(f"extracted to {dest}")
        return

    if not volume_exists(a.volume):
        docker(["volume", "create", a.volume], quiet=True)
        print(f"created volume {a.volume}")
    shell = f"tar xzf /export/{f.name} -C /out && ls /out"
    run_in_container(a.volume, f.parent, shell, volume_ro=False, host_ro=True, as_user=False)
    print(f"imported {f.name} into volume {a.volume}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    l = sub.add_parser("list", help="list the repacking_k_* folders in the volume")
    l.add_argument("--volume", default=_CFG["volume"])
    l.set_defaults(func=cmd_list)

    e = sub.add_parser("export")
    e.add_argument("--what", choices=["split", "all", "folder", "sets", "extras", "data"], default="split")
    e.add_argument("--path", nargs="+", help="folder(s) in the volume, * allowed (for --what folder)")
    d = _CFG["defaults"]
    e.add_argument("--datasets", default=None, help="for --what sets/extras: comma-separated, default all in the volume")
    e.add_argument("--modes", default=d["modes"], help="for --what sets/extras")
    e.add_argument("--algos", default=d["algos"], help="for --what sets/extras")
    e.add_argument("--k_list", default=d["k_list"], help="for --what sets/extras")
    e.add_argument("--parts", default="index,preview", help="for --what extras: index, preview or index,preview")
    e.add_argument("--move", action="store_true",
                   help="folder/sets: after the copy is verified (file count + size), delete those repacking_k_* from the volume")
    e.add_argument("--dest", default=str(config.resolve(_CFG["export_dir"])))
    e.add_argument("--name", default=None, help="archive file name (split / all)")
    e.add_argument("--volume", default=_CFG["volume"])
    e.set_defaults(func=cmd_export)

    i = sub.add_parser("import")
    i.add_argument("file", help="a .tar.gz made by 'export'")
    i.add_argument("--volume", default=_CFG["volume"])
    i.add_argument("--to-dir", default=None, help="extract on the host instead of the docker volume")
    i.set_defaults(func=cmd_import)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
