# train_v2_pm2.py
# Session wrapper around train_v2.py: remembers whether a training session is running / done, so it is never
# started twice, keeps a text log of the training output, and can be run under pm2, tmux or by hand.
#
#   python train_v2_pm2.py [train_v2 options]        start the session (or do nothing if it is running / done)
#   python train_v2_pm2.py --session-status          state of the session + the per-fold table of train_v2 --status
#   python train_v2_pm2.py --session-follow          show the end of the log and keep following it (Ctrl+C only leaves)
#   python train_v2_pm2.py --session-clear           forget the session (state + log); refuses while it is running
#   python train_v2_pm2.py --session-force           run again even if the session is "done" (finished folds are still skipped)
#   --session-name NAME   several independent sessions (default "default")
#
# Everything else is passed to train_v2.py, e.g.  python train_v2_pm2.py --only "iris_*" --epochs 300 --device 1
# train_v2 options --status / --clear / --dry-run are passed straight through without touching the session.
#
# Rules at start:  running (process alive + fresh heartbeat) -> exit 3, nothing started
#                  done                                     -> exit 0, nothing started (use --session-force)
#                  failed / stopped / running-but-dead      -> train_v2 runs again and continues per fold
# Exit code: the one of train_v2 (0 done, 1 some fold failed, 130 stopped), 3 = already running.
#
# pm2 example (check `pm2 start --help` for your pm2 version):
#   pm2 start train_v2_pm2.py --name strip-train --interpreter ~/venv-yolo/bin/python \
#       --stop-exit-codes 0 3 --max-restarts 5 --restart-delay 10000 --time
#   -> restarts after a crash or a reboot (pm2 save + pm2 startup), stops when the session is done.
#
# Session files: <session-dir>/<name>/session.json and train.log (default session-dir: ./sessions)
# Log: one line per finished progress bar (carriage-return updates are collapsed), plus a header per run.

import argparse
import codecs
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN = HERE / "train_v2.py"
HEARTBEAT_SEC = 15
PASS_THROUGH_ONLY = ("--status", "--clear", "--dry-run")

_lock = threading.Lock()


def now():
    return datetime.now().isoformat(timespec="seconds")


def pid_alive(pid):
    if not pid:
        return False
    if os.name == "nt":  # os.kill(pid, 0) would terminate the process on Windows, so ask the OS instead
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def session_paths(a):
    sdir = Path(a.session_dir).expanduser() / a.session_name
    return sdir, sdir / "session.json", sdir / "train.log"


def load_session(path):
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def save_session(path, sess):
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(sess, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)  # readers never see a half-written file


def is_running(sess, stale_sec):
    if not sess or sess.get("state") != "running" or not pid_alive(sess.get("pid")):
        return False
    try:
        age = (datetime.now() - datetime.fromisoformat(sess["heartbeat"])).total_seconds()
    except (KeyError, ValueError):
        return False
    return age < stale_sec  # a recycled pid after a reboot has no fresh heartbeat


def tail_lines(path, n):
    try:
        with open(path, "rb") as f:
            data = f.read()[-20000:]
    except OSError:
        return []
    return data.decode("utf-8", errors="replace").splitlines()[-n:]


def cmd_status(a, rest):
    sdir, spath, lpath = session_paths(a)
    sess = load_session(spath)
    if not sess:
        print(f"session '{a.session_name}': none (never started, or cleared)")
    else:
        alive = is_running(sess, a.session_stale_sec)
        state = sess["state"] + (" (process is gone: will run again)" if sess["state"] == "running" and not alive else "")
        print(f"session '{a.session_name}': {state}")
        for k in ("started_at", "finished_at", "exit_code", "runs", "pid", "child_pid", "heartbeat"):
            print(f"  {k}: {sess.get(k)}")
        print(f"  args: {' '.join(sess.get('args', []))}")
        print(f"  log: {lpath}")
        for line in tail_lines(lpath, 3):
            print(f"    | {line[:150]}")
    print()
    sys.stdout.flush()
    return subprocess.call([sys.executable, str(TRAIN), "--status", *rest])


def cmd_follow(a):
    sdir, spath, lpath = session_paths(a)
    if not lpath.is_file():
        sys.exit(f"no log yet: {lpath}")
    for line in tail_lines(lpath, 20):
        print(line)
    sys.stdout.flush()
    try:
        with open(lpath, "rb") as f:
            f.seek(0, os.SEEK_END)
            idle = 0
            while True:
                chunk = f.read()
                if chunk:
                    sys.stdout.write(chunk.decode("utf-8", errors="replace"))
                    sys.stdout.flush()
                    idle = 0
                    continue
                time.sleep(1)
                idle += 1
                if idle % 5 == 0 and not is_running(load_session(spath), a.session_stale_sec):
                    print(f"\n[session is not running: {(load_session(spath) or {}).get('state')}]")
                    return 0
    except KeyboardInterrupt:
        print("\n(left the log; training keeps running)")
        return 0


def cmd_clear(a):
    sdir, spath, lpath = session_paths(a)
    sess = load_session(spath)
    if is_running(sess, a.session_stale_sec) and not a.session_force:
        sys.exit("session is running; stop it first (pm2 stop / Ctrl+C) or add --session-force")
    if sdir.exists():
        try:
            shutil.rmtree(sdir)
        except PermissionError as e:
            sys.exit(f"cannot clear: files are in use ({e.filename}). Is the session still running? "
                     f"Check: python train_v2_pm2.py --session-status")
        print(f"session '{a.session_name}' cleared (the training results/manifest are not touched; "
              f"use: python train_v2.py --clear)")
    else:
        print("nothing to clear")
    return 0


def run_session(a, rest):
    sdir, spath, lpath = session_paths(a)
    sess = load_session(spath)
    if sess:
        if is_running(sess, a.session_stale_sec):
            print(f"session '{a.session_name}' is already running (pid {sess['pid']}, heartbeat {sess['heartbeat']}); "
                  f"not starting again. Watch it: python train_v2_pm2.py --session-follow")
            return 3
        if sess["state"] == "done" and not a.session_force:
            print(f"session '{a.session_name}' is already done ({sess.get('finished_at')}). "
                  f"Nothing to do; use --session-force to run again or --session-clear to forget it.")
            return 0
        print(f"session '{a.session_name}' was '{sess['state']}': running train_v2 again (finished folds are skipped)")

    sess = {"name": a.session_name, "state": "running", "pid": os.getpid(), "child_pid": None, "started_at": now(),
            "heartbeat": now(), "finished_at": None, "exit_code": None, "args": rest,
            "runs": (sess or {}).get("runs", 0) + 1}
    save_session(spath, sess)

    stop = threading.Event()

    def heartbeat():
        while not stop.wait(a.session_heartbeat_sec):
            sess["heartbeat"] = now()
            save_session(spath, sess)

    threading.Thread(target=heartbeat, daemon=True).start()

    env = dict(os.environ, PYTHONUNBUFFERED="1")
    proc = subprocess.Popen([sys.executable, "-u", str(TRAIN), *rest], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    sess["child_pid"] = proc.pid
    save_session(spath, sess)

    stopped = {"flag": False}

    def on_term(signum, frame):
        stopped["flag"] = True
        proc.terminate()

    signal.signal(signal.SIGTERM, on_term)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, on_term)

    tty = sys.stdout.isatty()
    dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
    log = open(lpath, "a", encoding="utf-8", buffering=1)
    log.write(f"\n=== run {sess['runs']} started {sess['started_at']}  train_v2 {' '.join(rest)}\n")

    buf, cr = "", False

    def emit(line):
        log.write(line + "\n")
        if not tty:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

    try:
        for chunk in iter(lambda: proc.stdout.read1(4096), b""):
            text = dec.decode(chunk)
            if tty:  # live progress bars exactly as ultralytics draws them
                sys.stdout.write(text)
                sys.stdout.flush()
            for ch in text:
                if ch == "\r":
                    cr = True
                elif ch == "\n":
                    emit(buf)
                    buf, cr = "", False
                else:
                    if cr:
                        buf, cr = "", False  # a carriage return rewrites the line: keep only the last version
                    buf += ch
        if buf:
            emit(buf)
        code = proc.wait()
    except KeyboardInterrupt:
        stopped["flag"] = True
        try:
            code = proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.terminate()
            code = proc.wait()
    finally:
        stop.set()

    if stopped["flag"] or code in (130, -2, -15):
        state, code = "stopped", 130
    else:
        state = "done" if code == 0 else "failed"
    sess.update(state=state, exit_code=code, finished_at=now(), heartbeat=now())
    save_session(spath, sess)
    log.write(f"=== run {sess['runs']} {state} (exit {code}) {sess['finished_at']}\n")
    log.close()
    print(f"[session '{a.session_name}': {state}, exit code {code}]")
    return code


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session-name", default="default")
    ap.add_argument("--session-dir", default=str(HERE / "sessions"))
    ap.add_argument("--session-status", action="store_true")
    ap.add_argument("--session-follow", action="store_true")
    ap.add_argument("--session-clear", action="store_true")
    ap.add_argument("--session-force", action="store_true")
    ap.add_argument("--session-stale-sec", type=int, default=120,
                    help="a 'running' session whose heartbeat is older than this is treated as dead (must be >= 3x heartbeat)")
    ap.add_argument("--session-heartbeat-sec", type=float, default=HEARTBEAT_SEC, help="how often a running session refreshes its heartbeat")
    a, rest = ap.parse_known_args()
    if a.session_stale_sec < 3 * a.session_heartbeat_sec:
        sys.exit(f"--session-stale-sec ({a.session_stale_sec}) must be at least 3x --session-heartbeat-sec "
                 f"({a.session_heartbeat_sec}); otherwise a running session looks dead and a second one would start")

    if a.session_status:
        sys.exit(cmd_status(a, rest))
    if a.session_follow:
        sys.exit(cmd_follow(a))
    if a.session_clear:
        sys.exit(cmd_clear(a))
    if any(x in rest for x in PASS_THROUGH_ONLY):
        sys.exit(subprocess.call([sys.executable, str(TRAIN), *rest]))
    sys.exit(run_session(a, rest))


if __name__ == "__main__":
    main()
