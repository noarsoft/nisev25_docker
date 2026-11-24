# Manual: gen strip fold (algo 1, 2)

ขอบเขต: ตั้งแต่ CSV ต้นฉบับ จนได้รูป fold (training / validation / test) ของ algo 1 (เทา) และ algo 2 (สี)
นอกขอบเขต: การเทรน YOLO, การนำรูปไปเทรนบน Windows (ทำแยก)

## 1. ไฟล์ในชุดนี้

| ไฟล์ | หน้าที่ |
|---|---|
| `init_data.py` | copy CSV ของ dataset เข้า `data/<dataset>/` |
| `gen_folds.py` | `split` แบ่ง fold เป็น CSV, `gen` สร้างรูป pack จาก CSV |
| `clear.py` | ล้างข้อมูล (host / Docker volume) |
| `lib.py`, `strip_pre2.py` | โลจิกเดิม (`gen_folds.py` เรียกใช้) |
| `Dockerfile`, `docker-compose.yml`, `requirements-gen.txt`, `.dockerignore` | สภาพแวดล้อม Docker (CPU) |

Dataset ที่รองรับ 12 ชุด: `iris, wine, breast_cancer, automobile, ionosphere` และ
`amphibians_{green_frogs, brown_frogs, common_toad, fire_bellied_toad, tree_frog, common_newt, great_crested_newt}`
ยังไม่รองรับ: multi-label (กลุ่ม 11 โครงการ), และชุดที่ไม่มี CSV (`adult, electric, student, hcc`)

## 2. หลักการ

```
data/<ds>/<ds>.csv --split--> split/ (CSV ราย fold + manifest.json) --gen--> รูป fold + manifest.json
```

- **`split/` คือตัวกำหนดข้อมูลทั้งหมด** (ขนาดเล็ก) รูปสร้างซ้ำได้จาก `split/` เสมอ
- ไม่มีรูปเดี่ยวระหว่างทาง: tile ถูกสร้างใน memory แล้ว pack เลย
- ผลของ algo 1 ตรงกับ pipeline เดิมทุก byte (ทดสอบกับ iris) และ algo 2 ตรงกับ `lib.strips_encode2` ระดับ tile
- ผลที่ gen จาก split CSV ชุดเดียวกัน ได้รูปเหมือนกันทุก byte ทั้งบน Windows และใน Docker (Linux)

โครงสร้างผลลัพธ์ใต้ `<out_dir>/<dataset>/`:

```
split/manifest.json
split/<nopca|pca>/fold_<i>/{train,val,test}.csv
<nopca|pca>/algo<1|2>/manifest.json                 # Ls/Hs ราย fold, จำนวนรูป, split_id, พารามิเตอร์
<nopca|pca>/algo<1|2>/repacking_k_<k>/<fold>/{training,validation,test}/{images,labels}
<nopca|pca>/algo<1|2>/repacking_k_<k>/<fold>/dataset.yaml
```

ที่เก็บข้อมูลเมื่อใช้ Docker
- `data/` บน host (mount เข้า `/data`) : CSV ต้นฉบับ
- named volume `stripout` (= `/out`) : ผลทั้งหมด ไม่เห็นในโฟลเดอร์ `out/` ของ host
- `out/` บน host : ใช้รับไฟล์ที่ส่งออกเท่านั้น

> ใช้ named volume เพราะเร็วกว่า bind mount ราว 30 เท่า (iris: 10 วินาที เทียบกับ 297 วินาที)

## 3. คำสั่ง

รันในโฟลเดอร์ `ultralytics` ตัวอย่างเป็น cmd (PowerShell: `${PWD}`, `$DS` / Linux: `$PWD`, `$DS`)

```
set DS=iris,wine,breast_cancer,automobile,ionosphere,amphibians_green_frogs,amphibians_brown_frogs,amphibians_common_toad,amphibians_fire_bellied_toad,amphibians_tree_frog,amphibians_common_newt,amphibians_great_crested_newt
```

### 3.1 รันผ่าน Docker

```
python init_data.py --all
docker build -t strip-gen .
docker compose run --rm strip split --datasets %DS% --dataset_root /data --out_dir /out --seed 42
docker compose run --rm strip gen   --datasets %DS% --out_dir /out --modes nopca,pca --algos 1,2 --k_list 0,2,4,8
```

### 3.2 รันตรง (ไม่ใช้ Docker)

```
pip install -r requirements-gen.txt
set PYTHONUTF8=1                                  (Linux: export PYTHONUTF8=1)
python init_data.py --all
python gen_folds.py split --datasets %DS% --dataset_root data --out_dir out --seed 42
python gen_folds.py gen   --datasets %DS% --out_dir out --modes nopca,pca --algos 1,2 --k_list 0,2,4,8
```
บน Windows เขียนไฟล์ลง NTFS ช้ากว่า volume ของ Docker (iris 4 combo k=0,2,4 ประมาณ 46 วินาที เทียบกับ 10 วินาที)

### 3.3 ตัวเลือก

`split`
- `--seed N` (ค่าเริ่มต้น 42) / `--overwrite` ถ้า `split/` มีอยู่แล้ว
- `--from_json_dir <โฟลเดอร์>` replay split จาก `output_kfold_data_<ds>.json` เดิม

`gen`
- `--modes nopca,pca` `--algos 1,2` `--k_list 0,2,4,8`
- `--seed N` seed สับลำดับตอน pack (k>=2) ค่าเริ่มต้น 0
- `--cell auto` ขนาด tile, `--no_jpeg_roundtrip` ข้ามขั้นจำลอง JPEG ของ tile
- `gen` ลบและเขียนทับเฉพาะ `repacking_k_<k>` ของ k ที่สั่ง k อื่นไม่ถูกแตะ

## 4. ตรวจผล

```
docker run --rm -v stripout:/out alpine ls /out/iris/nopca/algo1
```
หรือ Docker Desktop -> Volumes -> `stripout` -> Files

## 5. ส่งออก (copy ออกมา host)

> แนะนำใช้ `transfer.py` (สคริปต์เดียว Windows/Linux ไม่ต้องจำ `%cd%` / `$PWD`):
> ```
> python transfer.py export                          # split -> out/split_backup.tar.gz
> python transfer.py export --what all               # ทั้งหมด -> out/strip_all.tar.gz
> python transfer.py export --what folder --path iris/nopca/algo1/repacking_k_8   # โฟลเดอร์เดียว ไม่บีบ
> python transfer.py export --what data              # ./data -> out/data_backup.tar.gz
> python transfer.py import out/split_backup.tar.gz  # แตกลง volume stripout
> python transfer.py import out/split_backup.tar.gz --to-dir out   # แตกบน host (ไม่ใช้ docker)
> ```
> `--dest` เปลี่ยนโฟลเดอร์ปลายทาง, `--volume` เปลี่ยนชื่อ volume, `--name` เปลี่ยนชื่อไฟล์
> บน Linux ไฟล์ที่ส่งออกจะเป็นของผู้ใช้ปัจจุบัน (ไม่ใช่ root)

ตารางด้านล่างคือคำสั่ง docker แบบมือ (ผลเหมือนกัน)

| ต้องการ | คำสั่ง |
|---|---|
| เฉพาะ split (เล็ก ใช้ย้ายเครื่อง) | `docker run --rm -v stripout:/out:ro -v "%cd%\out:/export" alpine sh -c "cd /out && tar czf /export/split_backup.tar.gz */split"` |
| `data/` (CSV ต้นฉบับ) | `tar -czf out\data_backup.tar.gz data` |
| ทั้งหมดรวมรูปเป็นไฟล์เดียว | `docker run --rm -v stripout:/out:ro -v "%cd%\out:/export" alpine tar czf /export/strip_all.tar.gz -C /out .` |
| ทุกอย่างเป็นโฟลเดอร์ (ช้า ใช้พื้นที่ 2 เท่า) | `docker run --rm -v stripout:/out:ro -v "%cd%\out:/export" alpine cp -r /out/. /export/` |
| เฉพาะชุดเดียว + manifest | `docker run --rm -v stripout:/out:ro -v "%cd%\out:/export" alpine sh -c "mkdir -p /export/iris/nopca/algo1 && cp -r /out/iris/nopca/algo1/. /export/iris/nopca/algo1/"` |

Linux: เปลี่ยน `%cd%\out` เป็น `$PWD/out`
ถ้าจะใช้รูปจาก tar.gz ต้องแตกก่อน (`tar -xzf <ไฟล์> -C <โฟลเดอร์>`)

## 6. นำเข้า / ย้ายไปเครื่องอื่น

ย้ายเฉพาะ `split_backup.tar.gz` แล้ว gen ที่ปลายทาง (ไม่ต้องใช้ `data/`)

```
mkdir out
tar -xzf split_backup.tar.gz -C out
python gen_folds.py gen --datasets iris --out_dir out --modes nopca,pca --algos 1,2 --k_list 0,2,4,8
```

ใช้ Docker:
```
docker run --rm -v stripout:/out -v "%cd%\out:/export:ro" alpine tar xzf /export/split_backup.tar.gz -C /out
docker compose run --rm strip gen --datasets %DS% --out_dir /out --modes nopca,pca --algos 1,2 --k_list 0,2,4,8
```
กู้ทั้งหมดจาก `strip_all.tar.gz`: ใช้ `tar xzf /export/strip_all.tar.gz -C /out` ในคำสั่ง docker แบบเดียวกัน

เงื่อนไขให้รูปเหมือนเดิม: ไลบรารีเวอร์ชันเดียวกัน (ใช้ image เดียวกัน) และ `--seed`, `--cell`, `--k_list`, `--no_jpeg_roundtrip` ตรงกัน (บันทึกใน `k_runs` ของ manifest แต่ `gen` ไม่อ่านกลับเอง)
ยังไม่ได้ยืนยันกับ CPU คนละสถาปัตยกรรม (x86 / ARM)

### 6.1 ย้ายมาเครื่อง Linux (bash)

ไวยากรณ์ต่างจาก cmd: `%cd%\out` -> `$PWD/out`, `set DS=...` -> `DS=...`, `%DS%` -> `$DS`

```
cd ~/yolo8_strip_25/ultralytics
mkdir -p out                                    # วาง split_backup.tar.gz ไว้ที่ out/
DS=iris,wine,breast_cancer,automobile,ionosphere,amphibians_green_frogs,amphibians_brown_frogs,amphibians_common_toad,amphibians_fire_bellied_toad,amphibians_tree_frog,amphibians_common_newt,amphibians_great_crested_newt

docker build -t strip-gen .
docker run --rm -v stripout:/out -v "$PWD/out:/export:ro" alpine tar xzf /export/split_backup.tar.gz -C /out
docker run --rm -v stripout:/out:ro alpine ls /out/iris/split
docker compose run --rm strip gen --datasets $DS --out_dir /out --modes nopca,pca --algos 1,2 --k_list 0,2,4,8
```

- ถ้า `permission denied` ให้เพิ่มผู้ใช้เข้ากลุ่ม `docker` หรือใช้ `sudo`
- ลบไฟล์ใน `out/` ที่เป็นของ root: `sudo rm -rf out` หรือใช้ `python clear.py --host`

## 7. ล้างข้อมูล (`clear.py`)

```
python clear.py --host                        # ลบ out/ และ data/ ข้างสคริปต์
python clear.py --volume                      # ลบ volume stripout ทั้งก้อน
python clear.py --volume-images               # เก็บ split ลบเฉพาะ nopca/pca
python clear.py --volume-datasets iris,wine   # ลบทั้ง dataset (รวม split)
python clear.py --volume-k 8                  # ลบ repacking_k_8 ทุก dataset/mode/algo
python clear.py --all                         # --host + --volume
```
`--dry-run` แสดงอย่างเดียว, `--yes` ข้ามการถามยืนยัน
ข้อควรระวัง: `--host` ลบ `out/` ทั้งโฟลเดอร์ รวมไฟล์สำรอง (`*.tar.gz`) ในนั้นด้วย ให้ย้ายไปที่อื่นก่อน

เริ่มใหม่ทั้งหมด:
```
python clear.py --all --yes
python init_data.py --all
docker build -t strip-gen .
(แล้วรัน split และ gen ตามข้อ 3.1)
```

## 8. ปัญหาที่พบบ่อย

| อาการ | สาเหตุ / วิธีแก้ |
|---|---|
| `Dockerfile: no such file` | ต้อง `cd` เข้าโฟลเดอร์ `ultralytics` ก่อน build |
| `${PWD}\data is not a valid Windows path` | `${PWD}` ใช้ได้เฉพาะ PowerShell ใน cmd ใช้ `%cd%` |
| `FileNotFoundError: .../split/manifest.json` | ยังไม่ได้รัน `split` หรือ volume ถูกล้างแล้ว |
| `split ... already exists` | เติม `--overwrite` |
| `UnicodeEncodeError` (รันตรงบน Windows) | `set PYTHONUTF8=1` (ใน Docker ตั้งไว้แล้ว) |
| `rm -rf /out/` ฟ้อง Device or resource busy | เป็นจุด mount ไฟล์ข้างในถูกลบแล้ว ใช้ `clear.py` แทน |
| `WARN volume "stripout" already exists but was not created by Docker Compose` | ไม่เป็นปัญหา (หายหลัง `docker volume rm stripout` แล้วรันใหม่) |
| มองไม่เห็นผล gen ใน `out/` | ผลอยู่ใน volume `stripout` ให้ส่งออกตามข้อ 5 |
| gen ช้ามากผ่าน Docker | อย่าใช้ bind mount `./out:/out` ใช้ named volume ตาม compose |
| Docker engine ไม่ขึ้น (`dockerDesktopLinuxEngine`) | เปิด Docker Desktop แล้วรอให้พร้อม |
| ลบไฟล์เป็นของ root บน Linux ไม่ได้ | `sudo rm -rf out` หรือรัน Docker ด้วย `--user $(id -u):$(id -g)` |

## 9. ข้อควรจำ / ข้อจำกัด

- split ใหม่ด้วย `--seed 42` ได้ fold เดิม แต่ค่า PCA อาจต่างระดับ ~1e-15 ข้ามเครื่อง จึงควรเก็บ `split_backup.tar.gz` ไว้ ไม่ควรสุ่ม split ใหม่
- PCA fit กับข้อมูลทั้งชุดรวม test (ตามโค้ดเดิม)
- tile ของทั้ง fold อยู่ใน memory ตอน gen ยังไม่ได้ทดสอบกับข้อมูลขนาดใหญ่ (เช่น adult)
- คำสั่งที่ยังไม่ได้ทดสอบครบทุกรูปแบบ: ส่งออก split ทุก dataset (`*/split`), นำเข้าข้อ 6, ย้ายแบบตรวจก่อนลบ ให้ลองกับ iris ก่อน
