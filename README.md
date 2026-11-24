# nisev25_docker

Pipeline ของงาน strip: ข้อมูลตาราง (CSV) → รูป strip → ชุดข้อมูล k-fold → เทรน YOLOv8 → ทำนาย/ประเมินผล

- **algo 1** = strip สีเทา, **algo 2** = strip สี
- **mode** `nopca` = ใช้ feature เดิม, `pca` = ใช้ PC1/PC2
- ชื่อกลุ่มที่เทรน = `<dataset>_<mode>_algo<a>_k<k>` เช่น `iris_nopca_algo1_k8` (แต่ละกลุ่มมี 5 fold)
- Dataset ที่รองรับ 12 ชุด: iris, wine, breast_cancer, automobile, ionosphere, amphibians_{green_frogs, brown_frogs, common_toad, fire_bellied_toad, tree_frog, common_newt, great_crested_newt}

คำสั่งทั้งหมดรันในโฟลเดอร์ `ultralytics/` (Windows ใช้ `python`, Ubuntu ใช้ `python3` หรือ `python` ใน venv)

## โครงสร้าง

| ไฟล์ | หน้าที่ |
|---|---|
| `config.json` / `config.py` | ค่ากลาง: ชื่อ volume/image, รายชื่อ dataset (`run_datasets`), ค่า default ของ gen |
| `init_data.py` | คัดลอก CSV ต้นฉบับเข้า `data/<dataset>/` |
| `gen_folds.py` | `split` แบ่ง fold เป็น CSV, `gen` สร้างรูป strip + pack + label (รันใน Docker) |
| `run_gen.py` | ตัวห่อ `docker run` สำหรับ split/gen/build |
| `transfer.py` | export/import ระหว่าง Docker volume กับโฟลเดอร์ `out/` และเครื่องอื่น |
| `clear.py` | ล้าง `out/`, `data/`, volume |
| `train_v2.py` + `train_config.json` | เทรนทุกกลุ่ม/ทุก fold ตาม config, บันทึก manifest, ข้ามที่เสร็จแล้ว |
| `train_v2_pm2.py` + `ecosystem.config.js` | รันเทรนยาวผ่าน PM2 พร้อมจำสถานะ session |
| `predict_v2.py` | ทำนายจาก CSV / รูป / test ของ fold |
| `predict_watch_v2.py` | ทำนายแบบค้างรอ (โหลดโมเดลครั้งเดียว) |
| `evaluate_v2.py` | วัดผลบน test ทุก fold: accuracy, balanced acc, macro-F1, confusion matrix |

ข้อมูลที่สร้างได้ใหม่ (`data/`, `out/`, `runs_strip/`, `*.pt` ฯลฯ) ไม่ถูก commit ดู `.gitignore` ยกเว้น `out/split_backup.tar.gz` ที่เก็บไว้เพื่อทำซ้ำ split เดิม

## ข้อกำหนด

- Docker (สำหรับ gen), Python 3
- Ubuntu ที่เทรน: PM2 (`npm i -g pm2`), venv ที่มี `ultralytics`
- ตอนประเมิน/ทำนาย ให้ใช้ Pillow เวอร์ชันเดียวกับ Docker image (`pip install "pillow==12.2.0"`) เพื่อให้รูปตรงกับที่ใช้เทรน

## Scenario 1: เริ่มใหม่ทั้งหมด (เครื่องต้นทาง)

```bash
python run_gen.py all              # init data + build image (ถ้ายังไม่มี) + split + gen
python run_gen.py gen --preview    # gen โดยมีรูป preview (กรอบ + row_id) ด้วย
```

- ผลอยู่ใน Docker volume `stripout` ไม่ได้อยู่ในโฟลเดอร์ `out/`
- gen เขียน `index/<part>.csv` ทุกครั้ง บอกว่าแต่ละกล่องในรูปมาจาก `row_id` ไหน ส่วน `preview/` เขียนเมื่อใส่ `--preview`
- `all` รัน `split` ก่อน ถ้า volume มี split อยู่แล้วจะหยุด (ไม่ทับ) ให้ใช้ `run_gen.py gen` แทน อย่าใส่ `--overwrite` ถ้าไม่ต้องการ split ใหม่

## Scenario 1.2: ย้ายไปเครื่องอื่น (เช่น Ubuntu)

เครื่องต้นทาง:

```bash
python transfer.py export          # -> out/split_backup.tar.gz (~0.5 MB)
```

copy ไฟล์นี้ (และโปรเจ็คทั้งโฟลเดอร์ รวม `config.json`) ไปวางที่ `ultralytics/out/` ของเครื่องปลายทาง

เครื่องปลายทาง:

```bash
cd ~/yolo8_strip_25/ultralytics
python3 clear.py --all --yes                      # ล้างของเดิม (ทำก่อนวาง tar)
# วาง out/split_backup.tar.gz แล้วตรวจ
ls -l out/split_backup.tar.gz
python3 transfer.py import out/split_backup.tar.gz
python3 run_gen.py build
python3 run_gen.py gen                            # สร้างรูป algo 1,2 x nopca,pca x k ตาม config.json
```

`gen` จาก split เดียวกัน + seed เดียวกัน + Pillow เวอร์ชันเดียวกัน ได้รูปเหมือนเดิมทุกไบต์

## Scenario 2: เทรน (Ubuntu)

ติดตั้งครั้งเดียว:

```bash
python3 -m venv ~/venv-yolo        # ถ้าฟ้อง ensurepip: sudo apt install -y python3-venv python3-pip
source ~/venv-yolo/bin/activate
pip install --upgrade pip
pip install ultralytics
```

ดึงชุดเทรนออกจาก Docker มาที่ `out/` (`--move` = ลบใน volume หลังตรวจว่าคัดลอกครบ):

```bash
python transfer.py export --what sets --algos 1,2 --k_list 8 --move
```

เทรนแบบปกติ (ค่าตั้งอยู่ใน `train_config.json`, option ใน command line ทับค่าในไฟล์):

```bash
python train_v2.py --datasets iris,wine --modes nopca --algos 1,2 --k_list 8 --folds 0,1
python train_v2.py --weights yolov8m.pt --epochs 1000 --batch 32 --workers 1 --device 0 --seed 7 --patience 50
python train_v2.py --extra fliplr=0.0 mosaic=0     # ส่งค่าให้ YOLO ตรงๆ (เช่น ปิด augmentation)
python train_v2.py --only "iris_*"                 # เฉพาะกลุ่มที่ชื่อตรง
python train_v2.py --status                        # ตารางสถานะ (mAP, เวลา start/stop/interval)
python train_v2.py --redo                          # เทรนใหม่แม้เสร็จแล้ว
python train_v2.py --clear                         # ลบผลเทรน + manifest ของกลุ่มที่เลือก
```

- fold ที่ `done` และค่าตั้งเหมือนเดิมจะถูกข้าม, fold ที่ค้าง/ล้มเหลวจะเทรนใหม่ตั้งแต่ epoch 1
- ถ้าแก้ค่าเทรนแล้ว fold ที่เสร็จแล้วเป็น STALE ต้องใช้ `--redo`
- `--device`: บนเครื่องที่ใช้อยู่ 0 = RTX 3090 (V100 ใช้ไม่ได้กับ torch ที่ลงไว้)

ผลลัพธ์ต่อกลุ่ม `runs_strip/<group>/`:

```
train_manifest.json      สถานะทุก fold, เวลา, best_epoch, mAP, path best.pt, split_id
train_config.used.json   config ที่ใช้จริง
fold_0/ ... fold_4/
   epochs.csv            เวลาต่อ epoch
   train/weights/best.pt, last.pt
   train/results.csv, results.png, confusion_matrix.png ...
```

### เทรนยาวผ่าน PM2

```bash
cd ~/yolo8_strip_25/ultralytics
mkdir -p logs
source ~/venv-yolo/bin/activate
pm2 delete strip-train                  # ถ้ารันอยู่
python train_v2.py --clear              # (ถ้าต้องการเริ่มใหม่หมด)
python train_v2_pm2.py --session-clear  # ล้าง session เก่า (ไม่แตะผลเทรน)
pm2 start ecosystem.config.js --only strip-train
pm2 logs strip-train                    # ดูสด (Ctrl+C ออกได้ เทรนไม่หยุด)
pm2 save && pm2 startup                 # เปิดเองหลังรีบูต (รันคำสั่ง sudo ที่พิมพ์ออกมา)
python train_v2_pm2.py --session-status # สถานะ session + ตาราง fold
```

อื่นๆ: `--session-follow` (ดู log), `--session-force` (รันซ้ำแม้ session จบแล้ว), `--session-name NAME` (แยกหลาย session สำหรับรันขนาน โดยต้องเลือกกลุ่มคนละชุดด้วย `--only`)

อย่าใช้ `pm2 delete all` เพราะจะลบแอปอื่นที่อยู่ใน PM2 เดียวกัน

## Scenario 3: วัดผลบน test (Ubuntu)

ดึง `test.csv` ของ split มาก่อน (ครั้งเดียว) ตรวจด้วย `ls out/wine/split/pca/fold_0`:

```bash
python transfer.py export --what folder --path "*/split"
```

```bash
python predict_v2.py --group iris_nopca_algo1_k8 --fold 0 --test     # fold เดียว
python evaluate_v2.py --group iris_nopca_algo1_k8                     # ทุก fold ของกลุ่ม
python evaluate_v2.py                                                 # ทุกกลุ่มที่เทรนแล้ว
python evaluate_v2.py --only "iris_*" --folds 0,1,2
```

ผลอยู่ที่ `predictions/eval_<group>.json` (มี confusion matrix รวมทุก fold) และ `predictions/eval_summary.csv`

- ค่า mAP ในตารางสถานะเทรนวัดบน validation ที่ใช้เลือก best.pt จึงสูงเกินจริง ผลจริงดูจาก evaluate (test)
- ใช้ balanced accuracy / macro-F1 คู่กับ accuracy เมื่อคลาสไม่สมดุล (เช่น amphibians)
- คลาสที่ตอบ = คลาสของกล่องที่ confidence สูงสุด (`--conf` ค่าเริ่มต้น 0.001 ไม่มีเกณฑ์ตัดสิน)

## Scenario 4: ใช้งานจริง (Ubuntu)

โหมด `pca` ใช้ทำนายจาก CSV ดิบไม่ได้ (ไม่ได้เก็บพารามิเตอร์ PCA) ต้องส่งค่า PC1, PC2

### 4.1 ส่งแล้วจบ

```bash
# 1. test_input.csv -> test_input.jpg (ไม่ใช้โมเดล)
python predict_v2.py --group iris_nopca_algo1_k8 --fold 0 --input test_input.csv --no-predict --save-images imgs
# 2. ทำนายจากรูป
python predict_v2.py --group iris_nopca_algo1_k8 --fold 0 --images imgs/test_input.jpg
```

ทำนายจาก CSV ตรงๆ (ขั้น 1+2 ในคำสั่งเดียว): `--input test_input.csv` โดยไม่ใส่ `--no-predict`

### 4.2 โมเดลค้างรอ แล้วค่อยส่งข้อมูล

```bash
python predict_watch_v2.py --group iris_nopca_algo1_k8 --fold 0
```

ใส่ไฟล์ CSV (คอลัมน์ feature เหมือน `--input` ของ `predict_v2.py`) หรือรูป `.jpg`/`.png` ลงที่ `watch/inbox/` ผลอยู่ที่ `watch/outbox/<ชื่อไฟล์>.result.json` (คลาสสูงสุด, confidence, คะแนนทุกคลาส, เวลาที่ใช้) ถ้าผิดพลาดได้ `<ชื่อ>.error.json` รันผ่าน PM2 ได้ด้วยแอป `strip-predict` ใน `ecosystem.config.js`

## Preview และ index (ย้อนรอยรูป pack กลับเป็นแถวใน CSV)

```
repacking_k_<k>/<fold>/index/{training,validation,test}.csv   image, line, row_id, class, cx, cy, w, h, kind
repacking_k_<k>/<fold>/preview/<part>/                        รูปที่วางกรอบ + row_id (เฉพาะ gen --preview)
```

ทั้งสองอยู่นอกโฟลเดอร์ `training/validation/test` เพื่อไม่ให้ YOLO อ่านเป็นข้อมูลเทรน ดึงออกจาก volume เฉพาะส่วนนี้:

```bash
python transfer.py export --what extras --parts preview          # หรือ index, หรือ index,preview
```

## ลำดับคำสั่ง transfer.py (`--what`)

| ค่า | ได้อะไร |
|---|---|
| `split` (ค่าเริ่มต้น) | `out/split_backup.tar.gz` (CSV แบ่ง fold + manifest) |
| `all` | ทั้ง volume -> `out/strip_all.tar.gz` |
| `folder --path ...` | โฟลเดอร์ที่ระบุ (รองรับ `*`) เป็นโฟลเดอร์จริงใน `out/` |
| `sets` | `repacking_k_*` ตาม `--datasets/--modes/--algos/--k_list` พร้อม manifest ของ algo (`--move` ลบใน volume หลังตรวจ) |
| `extras` | เฉพาะ `index/` และ `preview/` |
| `data` | โฟลเดอร์ `data/` -> `out/data_backup.tar.gz` |

