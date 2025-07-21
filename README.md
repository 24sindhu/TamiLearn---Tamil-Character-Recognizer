# TamiLearn---Tamil-Character-Recognizer

# 🅰️ Tamil Character Image Classification

A deep learning project to classify **156 Tamil script characters** using Convolutional Neural Networks (CNNs). This project leverages a custom dataset of handwritten Tamil characters, preprocesses the data, and trains a robust CNN model with regularization to improve generalization.

---

## 📌 Overview

This project focuses on building an image classification system capable of recognizing Tamil characters from grayscale images. The system performs:

* Image pre-processing and bounding box extraction
* Resizing and padding to a standard 128×128 format
* CNN model training with dropout and L2 regularization
* Model evaluation and prediction of unseen characters

---

## 📁 Dataset

* **Source**: [HPL Tamil Offline Character Dataset](https://lipitk.sourceforge.net/datasets/tamilchardata.htm)
* **Classes**: 156 distinct Tamil characters
* **Format**: TIFF grayscale images organized in folders per character class

---

## 🧠 Model Architecture

A custom **CNN** with:

* Multiple `Conv2D` and `MaxPooling2D` layers
* `BatchNormalization` and `Dropout` for regularization
* Dense layers with ReLU activation
* Final `softmax` output layer for 156-class classification

Performance:

* **Train Accuracy**: \~97.8%
* **Test Accuracy**: \~92.5%
* Post-regularization test accuracy: \~67.2% (for updated model)

---

## ⚙️ Preprocessing Steps

* Extract bounding box of each character
* Resize to 100x100, pad to 128x128
* Normalize pixel values
* One-hot encode labels

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy opencv-python tensorflow keras matplotlib pillow
```

### Clone the repo

```bash
git clone https://github.com/yourusername/TamilCharacterClassifier.git
cd TamilCharacterClassifier
```

### Train the model

1. Place dataset in `./tamil_dataset_offline/`
2. Run the training script:

```bash
python train_model.py
```

---

## 🖼️ Predict a Character

You can test a new image using:

```python
from utils import preprocess_image, tamilCharacterCode
from tensorflow.keras.models import load_model

model = load_model('tamilALLEzhuthukalKeras_Model.h5')
img = preprocess_image('path/to/image.png')
prediction = tamilCharacterCode[model.predict(img).argmax()]
print("Predicted Character:", prediction)
```

---

## 📊 Results

| Epoch            | Accuracy     | Loss   |
| ---------------- | ------------ | ------ |
| 20               | 92.5% (test) | 0.2477 |
| 20 (regularized) | 67.2% (test) | 1.53   |

---

## 🔧 Improvements Made

* Added L2 Regularization
* Increased Dropout (0.3 – 0.4)
* Applied Batch Normalization
* Reduced Conv2D filter sizes



