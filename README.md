# Ripeness Detection System using Deep Learning

A deep learning-based image classification system for detecting banana ripeness levels using CNN architectures like VGG16, ResNet50, and Vision Transformer. Includes Grad-CAM for interpretability.

---

## Overview

This project automates the classification of banana ripeness stages—**unripe**, **ripe**, and **overripe**—using deep learning and image preprocessing techniques. It leverages popular models and includes Grad-CAM visualization for explainable AI.

---

## System Architecture

### Process Breakdown:

### 1. **Data Preprocessing**

- **Resizing**: Standardizes image size.
- **Image sharpening**: Enhances visual details.
- **Gaussian blurring**: Reduces noise.
- **NumPy conversion**: Prepares image arrays for training.

### 2. **Data Augmentation**

- Enhances dataset variability:
    - **Rotating**
    - **Flipping**
    - **Zooming**

### 3. **Training Phase**

- Trains models using:
    - **CNN**
    - **VGG16**
    - **ResNet50**
    - **Vision Transformer (ViT)**
- Uses preprocessed and augmented training images.

### 4. **Testing Phase**

- Evaluates the model using the testing set.
- Computes performance metrics: Accuracy, Precision, Recall, F1-Score.

### 5. **Grad-CAM Analysis**

- Applies Grad-CAM for model interpretability.
- Highlights areas in the image that contributed to classification.

---

## Model Features

- **Transfer Learning**: Leverages pre-trained models for better performance on small datasets.
- **Multi-class Classification**: Supports multiple ripeness levels.
- **Grad-CAM**: Generates heatmaps to understand model focus.

---

## Getting Started

### Installation

```bash
git clone https://github.com/waseralkarim/Ripeness-Detection-System-DeepLearning.git
cd Ripeness-Detection-System-DeepLearning
pip install -r requirements.txt
```

---

## Training the Model

```bash
python src/train.py \
  --data-dir data/ \
  --arch vgg16 \
  --epochs 25

```

---

## Evaluating the Model

```bash
python src/evaluate.py --checkpoint path/to/model.pt

```

---

## Grad-CAM Visualization

```bash
python src/gradcam.py --image path/to/image.jpg --checkpoint path/to/model.pt
```

- Outputs heatmaps indicating where the model focused.

---

## Directory Structure

```
Ripeness-Detection-System-DeepLearning/
├── architecture.png
├── data/
├── notebooks/
├── src/
├── app/
├── requirements.txt
└── README.md

```

---

## Results

- **Validation Accuracy:** ~96–98% depending on architecture
- **Models:** VGG16 & ResNet50 performed best in experiments
- **Grad-CAM:** Revealed that models focus on black spots and texture for classification

---

## References

- ImageNet Pretrained Models
- Research on Fruit Ripeness Classification with Deep Learning
- Grad-CAM: Visual Explanations from Deep Networks

---

## License

MIT License. See `LICENSE` for details.
