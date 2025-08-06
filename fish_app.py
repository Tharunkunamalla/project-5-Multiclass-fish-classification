
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load your best saved model
MODEL_PATH = 'cnn_fish_model.h5'  # You can change to any of the pre-trained model files if preferred
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels - Update these based on your dataset class folder names
class_names = ['Fish1', 'Fish2', 'Fish3', 'Fish4', 'Fish5']  # Replace with actual fish species

# Image preprocessing function
def preprocess_image(image: Image.Image):
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Streamlit UI
st.title("🐟 Fish Species Classifier")
st.write("Upload a fish image and the model will predict its category.")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    img_array = preprocess_image(image)
    predictions = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(predictions)]
    confidence_scores = {class_names[i]: float(predictions[i]) for i in range(len(class_names))}

    st.markdown(f"### 🐠 Prediction: **{predicted_class}**")
    st.markdown("### 📊 Confidence Scores:")
    st.json(confidence_scores)
