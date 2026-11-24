# predict_watch_v2.py
# Long-running predictor: loads the YOLO model ONCE, then waits for files and answers each one.
#
#   python predict_watch_v2.py --group iris_nopca_algo1_k8 --fold 0
#       drop a csv (feature columns, like predict_v2.py --input) or a ready-made .jpg/.png into  watch/inbox/
#       -> the answer appears in  watch/outbox/<name>.result.json  (an error gives  <name>.error.json)
#       -> the input file is moved to watch/inbox/done/ (or watch/inbox/failed/)
#       -> one line per sample on the console: top class + confidence
#   python predict_watch_v2.py --status       is it alive, since when, how many requests, last request (reads watch/status.json)
#
# Copy the file into the inbox atomically (write it elsewhere, then move/rename it in) or give it a name ending in
# .tmp / .part while it is being written: files that are still changing (size/time changed in the last --settle-sec) are left alone.
# One instance per watch folder (a second start is refused while the first is alive).
# Stop with Ctrl+C / SIGTERM (pm2 stop): the request being processed is finished first.
# pm2: see the "strip-predict" app in ecosystem.config.js.

import argparse
import json
import os
import shutil
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import predict_v2 as pv
import train_v2
from train_v2_pm2 import pid_alive

HERE = Path(__file__).resolve().parent
IMG_EXT = (".jpg", ".jpeg", ".png")


def now():
    return datetime.now().isoformat(timespec="seconds")


def write_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)  # readers never see half a file


def status_age(st):
    try:
        return (datetime.now() - datetime.fromisoformat(st["heartbeat"])).total_seconds()
    except (KeyError, ValueError, TypeError):
        return 1e9


def cmd_status(watch):
    p = watch / "status.json"
    st = pv.load_json(p)
    if not st:
        print(f"no status file: {p} (never started)")
        return 1
    alive = st.get("state") == "running" and pid_alive(st.get("pid")) and status_age(st) < 60
    print(f"watcher: {'ALIVE' if alive else 'not running'}   (file says: {st.get('state')})")
    for k in ("group", "fold", "weights", "pid", "started_at", "heartbeat", "requests", "samples", "errors", "last_request", "last_finished_at", "last_ms"):
        print(f"  {k}: {st.get(k)}")
    return 0 if alive else 1


def ready(path, settle):
    """a file is ready when it is not a temp name and has not changed for `settle` seconds"""
    if path.suffix.lower() in (".tmp", ".part") or path.name.startswith("."):
        return False
    try:
        return time.time() - path.stat().st_mtime >= settle
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--watch-dir", default=str(HERE / "watch"), help="holds inbox/, outbox/ and status.json")
    ap.add_argument("--poll", type=float, default=1.0, help="seconds between looks at the inbox")
    ap.add_argument("--settle-sec", type=float, default=1.0, help="a file must be unchanged this long before it is read")
    ap.add_argument("--label-col", default=None, help="true-class column (default: from the split manifest if present)")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--device", default=None)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--config", default=str(train_v2.DEFAULT_CONFIG))
    ap.add_argument("--data-root", dest="data_root", default=None)
    ap.add_argument("--project-root", dest="project_root", default=None)
    a = ap.parse_args()

    watch = Path(a.watch_dir).expanduser()
    if a.status:
        sys.exit(cmd_status(watch))
    if not a.group:
        ap.error("--group is required")

    inbox, outbox = watch / "inbox", watch / "outbox"
    done_dir, failed_dir = inbox / "done", inbox / "failed"
    for d in (inbox, outbox, done_dir, failed_dir):
        d.mkdir(parents=True, exist_ok=True)

    old = pv.load_json(watch / "status.json")
    if old and old.get("state") == "running" and pid_alive(old.get("pid")) and status_age(old) < 60 and old.get("pid") != os.getpid():
        print(f"a watcher is already running on {watch} (pid {old['pid']}); not starting a second one")
        sys.exit(3)

    t0 = time.time()
    ctx = pv.load_context(a.group, a.fold, a.config, a.data_root, a.project_root, a.weights)
    label_col = a.label_col or ctx.label_col
    weights = pv.resolve_weights(ctx)
    from ultralytics import YOLO
    model = YOLO(str(weights))
    kw = {"imgsz": ctx.canvas, "conf": a.conf, "verbose": False}
    if a.device is not None:
        kw["device"] = a.device
    from PIL import Image
    model.predict([Image.new("RGB", (ctx.canvas, ctx.canvas), (255, 255, 255))], **kw)  # warm-up: first real request is not slow
    print(f"model ready in {time.time() - t0:.1f}s: {weights}\nwatching {inbox}  (answers in {outbox}; Ctrl+C to stop)", flush=True)

    st = {"state": "running", "pid": os.getpid(), "group": a.group, "fold": a.fold, "weights": str(weights),
          "started_at": now(), "heartbeat": now(), "requests": 0, "samples": 0, "errors": 0,
          "last_request": None, "last_finished_at": None, "last_ms": None, "classes": ctx.labels}
    write_json(watch / "status.json", st)

    stop = {"flag": False}

    def on_signal(signum, frame):
        stop["flag"] = True

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, on_signal)

    def process(path):
        """-> (result dict, error text)"""
        source = path.stem
        try:
            if path.suffix.lower() in IMG_EXT:
                items = [(path.name, Image.open(path).convert("RGB"), None, {})]
            else:
                items, _ = pv.rows_to_items(ctx, pv.read_csv(path), source, label_col, False, path.name)
            preds = [pv.class_scores(r, len(ctx.labels)) for r in model.predict([im for _, im, _, _ in items], **kw)]
        except SystemExit as e:  # the helpers report bad input with sys.exit(message)
            return None, str(e.code)
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"
        results = []
        for (name, _, truth, extra), sc in zip(items, preds):
            order = sorted(range(len(ctx.labels)), key=lambda j: -sc[j])
            top = order[0] if sc[order[0]] > 0 else None
            rec = {"image": name, **extra, "prediction": ctx.labels[top] if top is not None else None,
                   "confidence": round(sc[order[0]], 4) if top is not None else 0.0,
                   "scores": {ctx.labels[j]: round(sc[j], 4) for j in order}}
            if truth is not None and truth >= 0:
                rec["true"], rec["correct"] = ctx.labels[truth], top == truth
            results.append(rec)
        return results, None

    try:
        last_beat = 0.0
        while not stop["flag"]:
            todo = sorted((p for p in inbox.iterdir() if p.is_file() and (p.suffix.lower() == ".csv" or p.suffix.lower() in IMG_EXT)
                           and ready(p, a.settle_sec)), key=lambda p: p.stat().st_mtime)
            for path in todo:
                t1 = time.time()
                results, err = process(path)
                ms = round((time.time() - t1) * 1000)
                st["requests"] += 1
                if err:
                    st["errors"] += 1
                    write_json(outbox / f"{path.name}.error.json", {"request": path.name, "error": err, "finished_at": now()})
                    shutil.move(str(path), str(failed_dir / path.name))
                    print(f"{path.name}: ERROR {err}", flush=True)
                else:
                    st["samples"] += len(results)
                    write_json(outbox / f"{path.name}.result.json",
                               {"request": path.name, "group": a.group, "fold": a.fold, "ms": ms, "finished_at": now(), "results": results})
                    shutil.move(str(path), str(done_dir / path.name))
                    for r in results:
                        top = f"{r['prediction']}  (confidence {r['confidence']:.3f})" if r["prediction"] else "no detection"
                        print(f"{path.name} [{r['image']}]: {top}   ({ms} ms)", flush=True)
                st.update(last_request=path.name, last_finished_at=now(), last_ms=ms, heartbeat=now())
                write_json(watch / "status.json", st)
                if stop["flag"]:
                    break
            if time.time() - last_beat >= 10:
                st["heartbeat"] = now()
                write_json(watch / "status.json", st)
                last_beat = time.time()
            time.sleep(a.poll)
    finally:
        st.update(state="stopped", heartbeat=now())
        write_json(watch / "status.json", st)
        print("watcher stopped", flush=True)


if __name__ == "__main__":
    main()
