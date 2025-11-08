
# Fish Species Classification

This project is a **Fish Species Classification** system that uses **Deep Learning** models to classify images of fish and seafood into multiple categories.

## 📌 Project Overview
The goal of this project is to accurately classify fish species from given images using different **Convolutional Neural Networks (CNNs)** and **Transfer Learning** architectures.

# 📁 Project Structure
```
├── data/                          # Dataset (download from Google Drive)
│   ├── train/
│   ├── validation/
│   └── test/
|
├── trained_models/
│   ├── cnn_fish_model.h5          # Download link provided above
│   ├── MobileNetV2_fish_model.h5
│   └── ......
|
|   ├──train_fish_model.ipynb #main file
│   ├── fish_app.py #app file
|
├── README.md
└── requirements.txt
```
**Dataset** : https://drive.google.com/drive/folders/1iKdOs4slf3XvNWkeSfsszhPRggfJ2qEd
## 📂 Dataset : 
The dataset contains images of different fish species, including:
- Gilt-head bream
- Horse mackerel
- Red mullet
- Red sea bream
- Sea bass
- Shrimp
- Striped red mullet
- Trout
- Black sea sprat
- Bass

Images are preprocessed and augmented before feeding into the models.

## 🧠 Models Used
We implemented and trained the following models:
- **Custom CNN Model** (Built from scratch)
- **VGG16** (Transfer Learning)
- **ResNet50** (Transfer Learning)
- **MobileNetV2** (Transfer Learning)
- **InceptionV3** (Transfer Learning)
- **EfficientNetB0** (Transfer Learning)

### 🔹 CNN Model
The CNN model is built from scratch using:
- **Conv2D Layers** for feature extraction
- **MaxPooling2D Layers** for downsampling
- **Flatten + Dense Layers** for classification
- **Dropout Layers** to reduce overfitting
- **Softmax Activation** for multi-class classification

📥 **Download CNN Model**: [Click Here](https://drive.google.com/file/d/1RzysZ1XgzAgYsAAouk9dZqIM9EHPSYIo/view?usp=sharing)

## ⚙️ Installation
```bash
# Clone the repository
git clone https://github.com/Tharunkunamalla/project-5-Multiclass-fish-classification.git

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage
```python
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the model
model = tf.keras.models.load_model("cnn_fish_model.h5")

# Load an image
img = image.load_img("test_fish.jpg", target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0) / 255.0

# Predict
predictions = model.predict(x)
print(predictions)
```

## 📊 Results
The models were trained for multiple epochs and evaluated on a validation set. Transfer Learning models achieved higher accuracy compared to the custom CNN.

| Model          | Accuracy |
|---------------|----------|
| CNN (Custom)  | ~85%     |
| VGG16         | ~92%     |
| ResNet50      | ~94%     |
| MobileNetV2   | ~95%     |
| InceptionV3   | ~93%     |
| EfficientNetB0| ~91%     |

# Streamlit-app-link: 
