Classification-Confusion-Matrix-using-YOLO-v5

## Overview

This repository contains a high-performance implementation of the **YOLOv5** (You Only Look Once) object detection architecture in PyTorch. YOLOv5 is designed for speed, accuracy, and ease of use, making it ideal for both research and production environments.

This version supports:

* **Object Detection** (Standard bounding boxes)
* **Instance Segmentation** (Masking objects)
* **Classification**
* **Exporting** to various formats (ONNX, TensorRT, CoreML, TFLite)

---

## 🚀 Quick Start

### 1. Installation

Clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd yolov5-master
pip install -r requirements.txt

```

### 2. Inference with `detect.py`

Run detection on images, videos, or folders:

```bash
python detect.py --source data/images --weights yolov5s.pt --conf 0.25

```

*Sources include:* `0` (webcam), `path/to/image.jpg`, `path/to/video.mp4`, or `path/to/dir/`.

---

## 🛠️ Training Your Own Model

To train a model on your custom dataset:

1. **Prepare your data:** Organize your images and labels in the YOLO format.
2. **Create a `.yaml` file:** Define paths and class names (see `data/coco128.yaml` for an example).
3. **Run training:**

```bash
python train.py --img 640 --batch 16 --epochs 100 --data custom_data.yaml --weights yolov5s.pt

```

---

## 📊 Key Features

* **Pre-trained Checkpoints:** Easily download YOLOv5n (Nano), YOLOv5s (Small), YOLOv5m (Medium), YOLOv5l (Large), and YOLOv5x (Extra Large).
* **Multi-Platform Support:** Deploy on CPU, GPU, or mobile devices.
* **Augmentation Pipeline:** Built-in Mosaic, Mixup, and Albumentations support for robust training.
* **Validation:** Robust validation scripts to calculate mAP (mean Average Precision).

---

## 📂 Repository Structure

```text
├── data/               # Configuration files for datasets and hyperparameters
├── models/             # Architecture definitions (YOLOv5s, v5m, etc.)
├── utils/              # Helper functions (plotting, logging, augmentation)
├── segment/            # Scripts for instance segmentation
├── classify/           # Scripts for image classification
├── train.py            # Main training script
├── detect.py           # Main inference script
├── val.py              # Validation script
└── requirements.txt    # Python dependencies

```

---

## 📈 Performance

YOLOv5 is optimized for real-world applications.

| Model | Size (pixels) | mAP@0.5:0.95 | CPU Speed (ms) | GPU Speed (ms) | Params (M) |
| --- | --- | --- | --- | --- | --- |
| **YOLOv5n** | 640 | 28.0 | 45 | 6.3 | 1.9 |
| **YOLOv5s** | 640 | 37.4 | 98 | 6.4 | 7.2 |
| **YOLOv5m** | 640 | 45.4 | 224 | 8.2 | 21.2 |

---

## 📝 Requirements

* Python 3.8+
* PyTorch >= 1.7
* Additional requirements listed in `requirements.txt`.


