# nisev25_docker

Pipeline for the strip project: tabular data (CSV) -> strip images -> k-fold datasets -> YOLOv8 training -> prediction / evaluation.

- **algo 1** = grey strip, **algo 2** = colour strip
- **mode** `nopca` = original features, `pca` = PC1/PC2
- A trained group is named `<dataset>_<mode>_algo<a>_k<k>`, e.g. `iris_nopca_algo1_k8` (each group has 5 folds)
- 12 supported datasets: iris, wine, breast_cancer, automobile, ionosphere, amphibians_{green_frogs, brown_frogs, common_toad, fire_bellied_toad, tree_frog, common_newt, great_crested_newt}

All commands are run inside the `ultralytics/` folder (`python` on Windows; `python3`, or `python` inside the venv, on Ubuntu).

## Layout

| File | Purpose |
|---|---|
| `config.json` / `config.py` | Central config: volume/image names, dataset list (`run_datasets`), gen defaults |
| `init_data.py` | Copies the source CSVs into `data/<dataset>/` |
| `gen_folds.py` | `split` writes per-fold CSVs, `gen` builds strip images + packs + labels (runs in Docker) |
| `run_gen.py` | `docker run` wrapper for split / gen / build |
| `transfer.py` | Export / import between the Docker volume, the `out/` folder and other machines |
| `clear.py` | Clears `out/`, `data/` and the volume |
| `train_v2.py` + `train_config.json` | Trains every group / fold from the config, writes manifests, skips finished folds |
| `train_v2_pm2.py` + `ecosystem.config.js` | Long training runs under PM2 with session tracking |
| `predict_v2.py` | Predict from a CSV, images, or the test set of a fold |
| `predict_watch_v2.py` | Long-running predictor (model loaded once) |
| `evaluate_v2.py` | Test-set evaluation over all folds: accuracy, balanced accuracy, macro-F1, confusion matrix |

Anything that can be regenerated (`data/`, `out/`, `runs_strip/`, `*.pt`, ...) is not committed, see `.gitignore`. The exception is `out/split_backup.tar.gz`, kept so the exact same split can be reproduced.

## Requirements

- Docker (for gen), Python 3
- Training machine (Ubuntu): PM2 (`npm i -g pm2`) and a venv with `ultralytics`
- For evaluation / prediction use the same Pillow version as the Docker image (`pip install "pillow==12.2.0"`) so the images match the ones used for training

## Scenario 1: start from scratch (source machine)

```bash
python run_gen.py all              # init data + build image (if missing) + split + gen
python run_gen.py gen --preview    # gen with preview images (boxes + row_id drawn on)
```

- Results live in the Docker volume `stripout`, not in the `out/` folder.
- Every gen writes `index/<part>.csv`, which tells which `row_id` each box in an image comes from. `preview/` is written only with `--preview`.
- `all` runs `split` first. If the volume already has a split it stops (nothing is overwritten); use `run_gen.py gen` instead. Do not add `--overwrite` unless you really want a new split.

## Scenario 1.2: move to another machine (e.g. Ubuntu)

Source machine:

```bash
python transfer.py export          # -> out/split_backup.tar.gz (~0.5 MB)
```

Copy this file (and the whole project folder, including `config.json`) to `ultralytics/out/` on the target machine.

Target machine:

```bash
cd ~/yolo8_strip_25/ultralytics
python3 clear.py --all --yes                      # clear old data (do this before placing the tar)
# put out/split_backup.tar.gz in place, then check it
ls -l out/split_backup.tar.gz
python3 transfer.py import out/split_backup.tar.gz
python3 run_gen.py build
python3 run_gen.py gen                            # algo 1,2 x nopca,pca x k as set in config.json
```

`gen` from the same split + same seed + same Pillow version gives byte-identical images.

## Scenario 2: training (Ubuntu)

One-time setup:

```bash
python3 -m venv ~/venv-yolo        # if it complains about ensurepip: sudo apt install -y python3-venv python3-pip
source ~/venv-yolo/bin/activate
pip install --upgrade pip
pip install ultralytics
```

Move the training sets out of Docker into `out/` (`--move` deletes them from the volume after the copy is verified):

```bash
python transfer.py export --what sets --algos 1,2 --k_list 8 --move
```

Normal training (settings are in `train_config.json`; command-line options override the file):

```bash
python train_v2.py --datasets iris,wine --modes nopca --algos 1,2 --k_list 8 --folds 0,1
python train_v2.py --weights yolov8m.pt --epochs 1000 --batch 32 --workers 1 --device 0 --seed 7 --patience 50
python train_v2.py --extra fliplr=0.0 mosaic=0     # pass values straight to YOLO (e.g. turn augmentation off)
python train_v2.py --only "iris_*"                 # only groups whose name matches
python train_v2.py --status                        # status table (mAP, start / stop / interval)
python train_v2.py --redo                          # train again even if finished
python train_v2.py --clear                         # delete results + manifest of the selected groups
```

- A fold that is `done` with unchanged settings is skipped; a running / failed / interrupted fold is trained again from epoch 1.
- If you change the training settings, finished folds become STALE and need `--redo`.
- `--device`: on the current machine 0 = RTX 3090 (the V100 does not work with the installed torch).

Output per group, in `runs_strip/<group>/`:

```
train_manifest.json      status of every fold, times, best_epoch, mAP, best.pt path, split_id
train_config.used.json   the config that was actually used
fold_0/ ... fold_4/
   epochs.csv            seconds per epoch
   train/weights/best.pt, last.pt
   train/results.csv, results.png, confusion_matrix.png ...
```

### Long runs under PM2

```bash
cd ~/yolo8_strip_25/ultralytics
mkdir -p logs
source ~/venv-yolo/bin/activate
pm2 delete strip-train                  # if it is running
python train_v2.py --clear              # (only if you want to start over)
python train_v2_pm2.py --session-clear  # forget the old session (training results are untouched)
pm2 start ecosystem.config.js --only strip-train
pm2 logs strip-train                    # live log (Ctrl+C only leaves the view, training keeps running)
pm2 save && pm2 startup                 # start on boot (run the sudo command it prints)
python train_v2_pm2.py --session-status # session + per-fold table
```

Also: `--session-follow` (tail the log), `--session-force` (run again even if the session is done), `--session-name NAME` (separate sessions for parallel runs; the runs must select different groups via `--only`).

Do not use `pm2 delete all`: it also removes the other apps in the same PM2.

## Scenario 3: test evaluation (Ubuntu)

Fetch the split `test.csv` files first (once). Check with `ls out/wine/split/pca/fold_0`:

```bash
python transfer.py export --what folder --path "*/split"
```

```bash
python predict_v2.py --group iris_nopca_algo1_k8 --fold 0 --test     # one fold
python evaluate_v2.py --group iris_nopca_algo1_k8                     # all folds of a group
python evaluate_v2.py                                                 # every trained group
python evaluate_v2.py --only "iris_*" --folds 0,1,2
```

Results: `predictions/eval_<group>.json` (includes the confusion matrix summed over the folds) and `predictions/eval_summary.csv`.

- The mAP in the training status table is measured on the validation set that also picks best.pt, so it is optimistic. The real result is the evaluate (test) output.
- Read balanced accuracy / macro-F1 next to accuracy when classes are imbalanced (e.g. amphibians).
- The predicted class is the class of the most confident box. `--conf` defaults to 0.001, so there is no decision threshold.

## Scenario 4: real use (Ubuntu)

Mode `pca` cannot predict from raw CSV columns (the PCA parameters are not stored); pass the PC1, PC2 values instead.

### 4.1 Send once and finish

```bash
# 1. test_input.csv -> test_input.jpg (no model)
python predict_v2.py --group iris_nopca_algo1_k8 --fold 0 --input test_input.csv --no-predict --save-images imgs
# 2. predict from the image
python predict_v2.py --group iris_nopca_algo1_k8 --fold 0 --images imgs/test_input.jpg
```

To predict straight from the CSV (steps 1 + 2 in one command) use `--input test_input.csv` without `--no-predict`.

### 4.2 Keep the model running, send data later

```bash
python predict_watch_v2.py --group iris_nopca_algo1_k8 --fold 0
```

Drop a CSV (feature columns like `--input` of `predict_v2.py`) or a ready-made `.jpg` / `.png` into `watch/inbox/`. The result is written to `watch/outbox/<file name>.result.json` (top class, confidence, score of every class, time taken); on failure you get `<name>.error.json`. It can also run under PM2 as the `strip-predict` app in `ecosystem.config.js`.

## index and preview (tracing a packed image back to CSV rows)

```
repacking_k_<k>/<fold>/index/{training,validation,test}.csv   image, line, row_id, class, cx, cy, w, h, kind
repacking_k_<k>/<fold>/preview/<part>/                        images with boxes + row_id drawn on (gen --preview only)
```

Both sit outside the `training/validation/test` folders so YOLO never reads them as training data. To copy only these out of the volume:

```bash
python transfer.py export --what extras --parts preview          # or index, or index,preview
```

## transfer.py `--what` values

| Value | What you get |
|---|---|
| `split` (default) | `out/split_backup.tar.gz` (fold CSVs + manifest) |
| `all` | the whole volume -> `out/strip_all.tar.gz` |
| `folder --path ...` | the given folder(s) (`*` allowed) as real folders under `out/` |
| `sets` | `repacking_k_*` for `--datasets/--modes/--algos/--k_list` plus the algo manifest (`--move` deletes them from the volume after verifying) |
| `extras` | only `index/` and `preview/` |
| `data` | the `data/` folder -> `out/data_backup.tar.gz` |
