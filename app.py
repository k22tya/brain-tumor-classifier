# app.py
import streamlit as st
import numpy as np
from PIL import Image

# Dummy model for testing (replace this later with your real model)
class DummyModel:
    def predict(self, x):
        return [[0.1, 0.2, 0.3, 0.4]]  # Fake prediction output

model = DummyModel()

# Define class names (must match real model training order)
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# App title
st.title("🧠 Brain Tumor Classifier")
st.write("Upload an MRI image and I’ll try to guess the tumor type.")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((150, 150))
    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.markdown(f"### 🧬 Prediction: **{predicted_class}**")
