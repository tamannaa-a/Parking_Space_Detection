import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Smart Parking Detector", page_icon="🚗")
st.title("🚗 Smart Parking Status Detector")
st.write("Upload an image of a parking space to check if it's **Occupied** or **Empty**.")

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="smart_parking_model_int8.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMG_SIZE = (224, 224)  # Must match training

def predict_parking_image(image):
    img = image.resize(IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Quantize input (normalize to [0,1], then to uint8)
    input_scale, input_zero_point = input_details[0]['quantization']
    if input_scale > 0:
        img_array = img_array / 255.0
        img_array = img_array / input_scale + input_zero_point
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0][0]

    # Dequantize output if needed
    if output_details[0]['dtype'] == np.uint8:
        output_scale, output_zero_point = output_details[0]['quantization']
        output_data = output_scale * (output_data - output_zero_point)

    result = "Occupied" if output_data > 0.5 else "Empty"
    confidence = output_data if output_data > 0.5 else 1 - output_data
    return result, confidence

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    result, confidence = predict_parking_image(image)
    st.subheader(f"Prediction: {result}")
    st.write(f"Confidence: {confidence*100:.2f}%")
