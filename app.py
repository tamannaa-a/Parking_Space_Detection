import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Set the page config
st.set_page_config(page_title="Smart Parking Space Detection", layout="centered")

# Load the TFLite model
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="smart_parking_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define class names (customize as per your dataset)
class_names = ["Empty", "Occupied"]

# Streamlit UI
st.title("🚗 Smart Parking Space Detection")
st.markdown("Upload a parking lot image and the model will detect if the parking space is **occupied** or **empty**.")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    resized_img = image.resize((224, 224))

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocessing
    img_array = np.array(resized_img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    pred_index = np.argmax(output)
    confidence = np.max(output)

    # Display result
    st.subheader("🧠 Prediction:")
    st.success(f"**{class_names[pred_index]}** ({confidence*100:.2f}%)")
