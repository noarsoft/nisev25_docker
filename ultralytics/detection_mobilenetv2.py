# python detection_mobilenetv2.py --test_image C:\yolo8_strip\ultralytics\test_crop\qr_crop_1.png --weights mobilenetv2_fold0_best.pth

import argparse
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description="Predict image with MobileNetV2")
    parser.add_argument('--test_image', type=str, required=True, help='Path to single image')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights (.pth)')
    args = parser.parse_args()

    image_path = args.test_image
    model_weights_path = args.weights

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {'GPU' if device.type == 'cuda' else 'CPU'}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    num_classes = 3  # 🔧 ปรับให้ตรงกับโมเดลของคุณ
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        print(f"Predicted class index: {predicted.item()}")

if __name__ == '__main__':
    main()
