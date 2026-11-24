# object detection k-fold[0] (cmd) > python C:\yolo8_strip\ultralytics\detect_strip.py --weights C:\yolo8_strip\ultralytics\runs\train\exp\weights\best.pt
# object detection k-fold[1] (cmd) > python C:\yolo8_strip\ultralytics\detect_strip.py --weights C:\yolo8_strip\ultralytics\runs\train\exp2\weights\best.pt
# copy to detection:  C:\yolo8_strip\ultralytics\backup_beast_cancer_train_and_kfold_pca2\output_k_fold3\1\test\images\0.jpg  >  C:\yolo8_strip\ultralytics\detect_input\0.jpg


import os
import shutil

import cv2
import json
import torch
from ultralytics import YOLO
from pathlib import Path


# project_directory
_output_directory = os.path.dirname(os.path.abspath(__file__))

input_img_path = "detect_input"
output_img_crop_path = "detect_out_crop"
output_img_crop_label_path = "detect_out_crop_labe"
save_txt = True
save_crop = True
save_conf = True
hide_labels = False
hide_conf = False

input_img_path = os.path.join(_output_directory, "detect_input")
if os.path.exists(input_img_path):
    shutil.rmtree(input_img_path)
os.makedirs(input_img_path)

output_img_path = os.path.join(_output_directory, "detect_out")
if os.path.exists(output_img_path):
    shutil.rmtree(output_img_path)
os.mkdir(output_img_path)

output_img_crop_path = os.path.join(_output_directory, "detect_out_crop")
if os.path.exists(output_img_crop_path):
    shutil.rmtree(output_img_crop_path)
os.mkdir(output_img_crop_path)

output_img_crop_label_path = os.path.join(_output_directory, "detect_out_crop_label")
if os.path.exists(output_img_crop_label_path):
    shutil.rmtree(output_img_crop_label_path)
os.mkdir(output_img_crop_label_path)



def xyxy2xywh(x):
    # Convert bounding box from xyxy format to xywh
    y = x.clone() if isinstance(x, torch.Tensor) else torch.tensor(x)
    y[:, 2] = y[:, 2] - y[:, 0]  # w = x2 - x1
    y[:, 3] = y[:, 3] - y[:, 1]  # h = y2 - y1
    y[:, 0] = y[:, 0] + y[:, 2] / 2  # x center
    y[:, 1] = y[:, 1] + y[:, 3] / 2  # y center
    return y

def save_one_box(xyxy, im, file, BGR=True):
    x1, y1, x2, y2 = map(int, xyxy)
    cropped = im[y1:y2, x1:x2]
    cv2.imwrite(str(file), cropped)


model = YOLO("yolov8n.pt")
names = model.names

print(input_img_path)
while True:
    for file in os.listdir(input_img_path):
        if not file.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        img_path = os.path.join(input_img_path, file)
        im = cv2.imread(img_path)
        results = model(im)

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            i_crop = 0
            for box, conf, cls in zip(boxes, confidences, classes):
                c = int(cls)
                label = names[c]

                # Convert box to xywh
                xyxy_tensor = torch.tensor(box).view(1, 4)
                xywh = xyxy2xywh(xyxy_tensor).view(-1).tolist()

                # Write txt
                if save_txt:
                    txt_path = os.path.join(output_img_crop_label_path, Path(file).stem + ".txt")
                    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
                    line = (c, *xywh, conf) if save_conf else (c, *xywh)
                    with open(txt_path, "a") as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                # Save cropped image
                if save_crop:
                    crop_dir = os.path.join(output_img_crop_path, Path(file).stem)
                    os.makedirs(crop_dir, exist_ok=True)
                    crop_file = os.path.join(crop_dir, f"{Path(file).stem}_{i_crop}.jpg")
                    save_one_box(box, im, file=crop_file)

                # Write JSON
                json_dir = os.path.join(output_img_crop_label_path, Path(file).stem)
                os.makedirs(json_dir, exist_ok=True)
                result_pred = {
                    "label_id": int(c),
                    "label": label,
                    "xywh": [round(n, 2) for n in xywh],
                    "conf": f"{conf:.4f}"
                }
                json_file = os.path.join(json_dir, f"{Path(file).stem}_{i_crop}.json")
                with open(json_file, "w") as f:
                    json.dump(result_pred, f)

                i_crop += 1