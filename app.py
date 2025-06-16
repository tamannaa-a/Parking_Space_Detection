import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import gdown
import os

st.title("🚗 Smart Parking Space Allocation")

# Define model file
MODEL_FILE = "smart_parking_model.tflite"

# Google Drive file ID (replace this with your actual ID)
FILE_ID = "YOUR_GOOGLE_DRIVE_FILE_ID"

# Download model if not present
if not os.path.exists(MODEL_FILE):
    st.info("Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_FILE, quiet=False)

# Check model existence
if not os.path.exists(MODEL_FILE):
    st.error("❌ Model file not found. Please check the Google Drive link.")
    st.stop()

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_FILE)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Upload image
uploaded_file = st.file_uploader("Upload a parking lot image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    image_resized = image.resize((224, 224))  # Match your training size
    input_array = np.expand_dims(np.array(image_resized, dtype=np.float32) / 255.0, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    # Assume output is [free_prob, occupied_prob]
    class_names = ["Free", "Occupied"]
    prediction = np.argmax(output)

    st.subheader("📊 Prediction")
    st.success(f"Predicted Status: **{class_names[prediction]}**")
