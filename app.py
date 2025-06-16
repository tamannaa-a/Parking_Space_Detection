import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.title("🚗 Smart Parking Status Detector")
st.write("Upload an image of a parking space to check if it's *Occupied* or *Empty*.")

# Constants
MODEL_PATH = "smart_parking_model.tflite"
IMG_SIZE = (224, 224)

# Load TFLite model
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file not found at {MODEL_PATH}. Please ensure it is uploaded.")
else:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def predict_parking_image(image):
    img = image.resize(IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array.astype(np.float32), axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

    result = "Occupied" if prediction > 0.5 else "Empty"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return result, confidence

# Run prediction
if uploaded_file and os.path.exists(MODEL_PATH):
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    result, confidence = predict_parking_image(image)
    st.subheader(f"Prediction: {result}")
    st.write(f"Confidence: {confidence * 100:.2f}%")
