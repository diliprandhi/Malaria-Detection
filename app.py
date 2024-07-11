import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

st.title("Malaria Detection App")

def load_model():
    model = tf.keras.models.load_model('Models/my_model.keras')
    return model

def predict(image, model):
    image = cv2.resize(image, (64, 64))
    image = np.expand_dims(image, axis=0) / 255.0
    prediction = model.predict(image)
    return "Parasitized" if prediction > 0.5 else "Uninfected"

model = load_model()

uploaded_file = st.file_uploader("Choose a cell image...", type="png")
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, channels="BGR")
    result = predict(image, model)
    st.write(f"The cell is: {result}")