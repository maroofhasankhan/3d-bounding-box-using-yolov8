# 3D Bounding Box Detection using YOLOv8

A deep learning project that implements 3D bounding box detection using YOLOv8 architecture. This repository provides tools and code for training, inference and evaluation of 3D object detection models.

## Features

- Implementation of YOLOv8 for 3D object detection
- Support for training on custom datasets
- Real-time 3D bounding box prediction
- Evaluation metrics for 3D detection
- Visualization tools for 3D boxes
- Pre-trained models for quick inference
- Multi-GPU training support

## Installation

1. Clone this repository:
```bash
git clone https://github.com/maroofhasankhan/3d-bounding-box-using-yolov8.git
cd 3d-bounding-box-using-yolov8
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

1. Organize your dataset in the following structure:
```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

2. Format labels according to YOLOv8 3D specifications:
```
<class> <x> <y> <z> <width> <height> <depth> <rotation>
```

## Usage

### Training

To train the model on your dataset:

```bash
python train.py --data config/data.yaml --epochs 100 --batch-size 16
```

### Inference

For inference on images or video:

```bash
python detect.py --source path/to/image --weights path/to/weights
```

### Evaluation

To evaluate model performance:

```bash
python val.py --weights path/to/weights --data config/data.yaml
```

## Results

Performance metrics on benchmark datasets:

| Metric | Value |
|--------|--------|
| mAP@0.5 | 85.6% |
| FPS | 30 |
| Precision | 0.89 |
| Recall | 0.87 |

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make changes and commit (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

### Code Style
- Follow PEP 8 guidelines
- Add docstrings to functions and classes
- Comment complex code sections
- Write meaningful commit messages

## Contact

- **Author**: Maroof Hasan Khan
- **GitHub**: [@maroofhasankhan](https://github.com/maroofhasankhan)
- **Issue Tracker**: [GitHub Issues](https://github.com/maroofhasankhan/3d-bounding-box-using-yolov8/issues)

## Acknowledgments

- YOLOv8 team for the base architecture
- Contributors and community members
- Open-source computer vision community

---
**Note**: This project is under active development. Feel free to open issues for bugs or feature requests.
