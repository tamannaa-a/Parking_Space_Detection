import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Smart Parking Classifier", layout="centered")
st.title("🚗 Smart Parking Space Allocation")
st.write("Upload an image of a parking lot to check if it's occupied or free.")

uploaded_file = st.file_uploader("Choose a parking image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((224, 224))
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path="smart_parking_model.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess
    img_array = np.array(image).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    if input_details[0]['dtype'] == np.uint8:
        img_array = img_array / 255.0
        img_array = (img_array * 255).astype(np.uint8)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    class_names = ['Free', 'Occupied']
    prediction = class_names[np.argmax(output_data)]

    st.success(f"🚦 Prediction: **{prediction}**")
