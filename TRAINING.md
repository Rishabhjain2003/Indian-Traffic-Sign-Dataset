# Training a YOLOv8 Model for Indian Traffic Sign Detection

This guide walks you through training a custom YOLOv8 model on Indian traffic sign datasets so that the pipeline's detection stage has a working `best.pt` weights file.

---

## 1. Recommended Datasets

| Dataset | Kaggle Slug | Notes |
|---|---|---|
| Indian Traffic Sign Dataset | `pkdarabi/indian-traffic-sign-dataset` | ~40 classes, pre-labelled bounding boxes |
| Indian Traffic Signs Dataset | `manishkr1754/indian-traffic-signs-dataset` | Complementary set of sign images |
| Indian Road Signs | `namanmakkar/indian-road-signs` | Additional real-world sign images |

Download one (or combine several) using the Kaggle CLI:

```bash
pip install kaggle
kaggle datasets download -d pkdarabi/indian-traffic-sign-dataset
unzip indian-traffic-sign-dataset.zip -d data/
```

---

## 2. Convert to YOLOv8 Format

YOLOv8 expects the following folder structure:

```
data/
├── train/
│   ├── images/
│   │   ├── img_001.jpg
│   │   └── ...
│   └── labels/
│       ├── img_001.txt
│       └── ...
├── val/
│   ├── images/
│   └── labels/
└── dataset.yaml
```

Each label file is a plain text file with one line per object:

```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are **normalised** (0–1) relative to image dimensions.

> **Tip:** If your dataset already provides PASCAL VOC (XML) annotations,
> convert them with a script like:
>
> ```python
> # Convert VOC XML to YOLO format
> import xml.etree.ElementTree as ET
> from pathlib import Path
>
> def voc_to_yolo(xml_path, class_names, img_w, img_h):
>     tree = ET.parse(xml_path)
>     lines = []
>     for obj in tree.findall(".//object"):
>         cls = obj.find("name").text
>         if cls not in class_names:
>             continue
>         cls_id = class_names.index(cls)
>         box = obj.find("bndbox")
>         x1 = float(box.find("xmin").text)
>         y1 = float(box.find("ymin").text)
>         x2 = float(box.find("xmax").text)
>         y2 = float(box.find("ymax").text)
>         xc = ((x1 + x2) / 2) / img_w
>         yc = ((y1 + y2) / 2) / img_h
>         w = (x2 - x1) / img_w
>         h = (y2 - y1) / img_h
>         lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
>     return "\n".join(lines)
> ```

---

## 3. Create `dataset.yaml`

Create a `data/dataset.yaml` file that tells YOLOv8 where to find the data:

```yaml
# data/dataset.yaml
path: ./data          # root directory
train: train/images   # relative to 'path'
val: val/images       # relative to 'path'

# Class names — update these to match YOUR dataset
names:
  0: speed_limit_20
  1: speed_limit_30
  2: speed_limit_50
  3: speed_limit_60
  4: speed_limit_70
  5: speed_limit_80
  6: no_entry
  7: stop
  8: yield
  9: no_parking
  10: no_horn
  11: pedestrian_crossing
  12: school_zone
  13: left_turn
  14: right_turn
  15: roundabout
  16: one_way
  17: u_turn_prohibited
  18: road_work
  19: speed_breaker
  # ... add more classes as needed
```

---

## 4. Train the Model

Install `ultralytics` if you haven't already:

```bash
pip install ultralytics
```

Run training:

```bash
yolo task=detect mode=train \
    model=yolov8n.pt \
    data=data/dataset.yaml \
    epochs=50 \
    imgsz=640 \
    batch=16 \
    patience=10 \
    name=indian_traffic_signs
```

### Model Size Options

| Model | Params | mAP (COCO) | Speed | Use Case |
|---|---|---|---|---|
| `yolov8n.pt` | 3.2M | 37.3 | Fastest | Quick prototyping / CPU |
| `yolov8s.pt` | 11.2M | 44.9 | Fast | Good balance |
| `yolov8m.pt` | 25.9M | 50.2 | Medium | Higher accuracy |
| `yolov8l.pt` | 43.7M | 52.9 | Slower | Production quality |
| `yolov8x.pt` | 68.2M | 53.9 | Slowest | Maximum accuracy |

Start with `yolov8n.pt` for fast iteration, then scale up as needed.

### GPU Training

If you have an NVIDIA GPU with CUDA:

```bash
yolo task=detect mode=train \
    model=yolov8n.pt \
    data=data/dataset.yaml \
    epochs=100 \
    imgsz=640 \
    batch=32 \
    device=0 \
    name=indian_traffic_signs
```

---

## 5. Evaluate the Model

After training completes, validate on the validation set:

```bash
yolo task=detect mode=val \
    model=runs/detect/indian_traffic_signs/weights/best.pt \
    data=data/dataset.yaml
```

This will output metrics including:
- **mAP@50** — mean average precision at IoU 0.5
- **mAP@50-95** — mAP across IoU thresholds 0.5–0.95
- **Precision / Recall** per class
- Confusion matrix

Aim for **mAP@50 ≥ 0.70** before using the model in the pipeline.

---

## 6. Export and Place the Weights

Copy the best weights to the project's `weights/` directory:

```bash
mkdir -p weights
cp runs/detect/indian_traffic_signs/weights/best.pt weights/best.pt
```

The pipeline will look for `weights/best.pt` by default (configurable via `YOLO_WEIGHTS` env var).

Verify the weights load correctly:

```python
from ultralytics import YOLO

model = YOLO("weights/best.pt")
print(f"Classes: {model.names}")
print(f"Total classes: {len(model.names)}")
```

---

## 7. Tips for Better Results

1. **Data augmentation**: YOLOv8 applies augmentation by default (mosaic, flip, HSV jitter). You can tune these in the training command.

2. **More data**: Combine multiple Kaggle datasets for broader coverage of Indian sign types.

3. **Class balance**: If some sign classes are under-represented, duplicate or augment those images.

4. **Transfer learning**: Starting from a COCO-pretrained model (`yolov8n.pt`) gives you a significant head start.

5. **Frame-level testing**: After training, test on actual dashcam frames to verify performance in real conditions:

   ```bash
   yolo task=detect mode=predict \
       model=weights/best.pt \
       source=output/frames/<video_id>/ \
       conf=0.35 \
       save=True
   ```

---

## Quick Reference

```bash
# Full workflow in 5 commands:

# 1. Download dataset
kaggle datasets download -d pkdarabi/indian-traffic-sign-dataset && unzip *.zip -d data/

# 2. Prepare dataset.yaml (see Section 3 above)

# 3. Train
yolo task=detect mode=train model=yolov8n.pt data=data/dataset.yaml epochs=50 imgsz=640

# 4. Evaluate
yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=data/dataset.yaml

# 5. Deploy
cp runs/detect/train/weights/best.pt weights/best.pt
```
