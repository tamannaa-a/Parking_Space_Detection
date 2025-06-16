%%writefile app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("smart_parking_model.h5")
img_size = (224, 224)

st.title("🚗 Smart Parking Status Detector")
st.write("Upload an image of a parking space to check if it's *Occupied* or *Empty*.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def predict_parking_image(image):
    img = image.resize(img_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    prediction = model.predict(img_array)[0][0]
    result = "Occupied" if prediction > 0.5 else "Empty"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return result, confidence

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    result, confidence = predict_parking_image(image)
    st.subheader(f"Prediction: {result}")
    st.write(f"Confidence: {confidence*100:.2f}%")
