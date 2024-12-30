import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

interpreter = tf.lite.Interpreter(model_path="mask_detection_modell.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

uploaded_file = st.file_uploader("Select an Image: ", type=["jpg", "jpeg", "png"])
result = ""
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    input_image = cv2.imdecode(file_bytes, 1)

    st.image(input_image, channels="BGR")
    input_image_resized = cv2.resize(input_image, (128, 128))
    input_image_scaled = input_image_resized / 255.0
    input_image_reshaped = np.reshape(input_image_scaled, [1, 128, 128, 3]).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_image_reshaped)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    input_pred_label = np.argmax(output_data)

    if input_pred_label == 1:
        result = 'The person in the image is wearing a mask'
    else:
        result = 'The person in the image is not wearing a mask'

st.success(result)
