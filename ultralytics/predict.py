from ultralytics import YOLO
import shutil
import json
import os
import sys
from pathlib import Path
import cv2

_weight = f'C:/yolo8_strip/ultralytics/runs/detect/train/weights/best.pt'



FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative





# project_directory
_output_directory = os.path.dirname(os.path.abspath(__file__))
conf_file = os.path.join(_output_directory, "detect.conf") #set 1 to loop runing
if not os.path.exists(conf_file):
    with open(conf_file + '.txt', 'a') as f:
        f.write('1')

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


# Load model
# Perform prediction on an image or directory
model = YOLO(_weight)

while True:
    # with open(conf_file, 'r') as f:
    #     status = f.read()
    # if status != "1":
    #     break
    for r, d, f in os.walk(input_img_path):
        if len(f)>0:
            print("file in coming")
        for file in f:

            # print(file)
            result_preds = []
            extension_allowed = '.jpg'
            ext_len = len(extension_allowed)
            strip = file[0:len(file) - ext_len]
            if os.path.exists(os.path.join(output_img_crop_path, strip)):
                shutil.rmtree(os.path.join(output_img_crop_path, strip))
            os.mkdir(os.path.join(output_img_crop_path, strip))
            if os.path.exists(os.path.join(output_img_crop_label_path, strip)):
                shutil.rmtree(os.path.join(output_img_crop_label_path, strip))
            os.mkdir(os.path.join(output_img_crop_label_path, strip))
            i_crop = 0
            source = os.path.join(input_img_path, file)

            results = model.predict(source=source, save=True, conf=0.25)

            for result in results:
                for box in result.boxes:
                    print(box)
                    class_id = int(box.cls)  # Class index
                    confidence = box.conf.item()  # Convert tensor to float
                    label = result.names[class_id]  # Class name
                    print(f"Class: {label}, Class_id: {class_id}, Confidence: {confidence:.2f}")

            # Load the original image using OpenCV
            image = cv2.imread(input_img_path)

            # Process each detection
            for i, result in enumerate(results):
                for j, box in enumerate(result.boxes):
                    # Get bounding box in xyxy format
                    xyxy = box.xyxy[0].tolist()  # [x_min, y_min, x_max, y_max]

                    # Convert to xywh format
                    x_center = (xyxy[0] + xyxy[2]) / 2  # (x_min + x_max) / 2
                    y_center = (xyxy[1] + xyxy[3]) / 2  # (y_min + y_max) / 2
                    width = xyxy[2] - xyxy[0]  # x_max - x_min
                    height = xyxy[3] - xyxy[1]  # y_max - y_min
                    print(f"Box {j} - xywh: ({x_center}, {y_center}, {width}, {height})")

                    # Crop the detected object from the image
                    x_min, y_min, x_max, y_max = map(int, xyxy)
                    cropped_image = image[y_min:y_max, x_min:x_max]

                    # Save the cropped image
                    save_path = os.path.join(output_img_crop_label_path, f'cropped_image_{i}_{j}.jpg')
                    cv2.imwrite(save_path, cropped_image)
                    print(f"Cropped image saved to {save_path}")

            os.remove(os.path.join(input_img_path, file))









