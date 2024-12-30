import streamlit as st
from tensorflow import keras
import numpy as np
import cv2

model = keras.models.load_model("mask_detection_model.h5")

uploaded_file = st.file_uploader("Select an Image: ", type=["jpg", "jpeg", "png"])

result = ""
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    input_image = cv2.imdecode(file_bytes, 1)

    st.image(input_image, channels="BGR")
    input_image_resized = cv2.resize(input_image, (128,128))
    input_image_scaled = input_image_resized/255
    input_image_reshaped = np.reshape(input_image_scaled, [1,128,128,3])

    input_prediction = model.predict(input_image_reshaped)
    input_pred_label = np.argmax(input_prediction)

    if input_pred_label == 1:
        result = 'The person in the image is wearing a mask'
    else:
        result = 'The person in the image is not wearing a mask'

st.success(result)

