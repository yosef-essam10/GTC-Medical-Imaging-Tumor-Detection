%%writefile app.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("brain_tumor_model.keras")

# Streamlit app design
st.set_page_config(page_title="Brain Tumor Detector", page_icon="🧠", layout="centered")

st.markdown(
    """
    <style>
        body {
            background-color: black;
            color: white;
        }
        .stApp {
            background-color: #000000;
        }
        h1, h2, h3 {
            color: #00FFAA;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧠 Brain Tumor Detection App")
st.write("Upload an MRI image and let the AI model predict if a brain tumor is present.")

uploaded_file = st.file_uploader("📤 Upload an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    st.image(uploaded_file, caption="Uploaded MRI Image", use_column_width=True)

    img = image.load_img(uploaded_file, target_size=(150, 150))
    img_array = image.img_to_array(img)

    if img_array.shape[-1] == 1:
        img_array = np.repeat(img_array, 3, axis=-1)

    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        st.markdown("<h2 style='color: red;'>⚠️ Brain Tumor Detected</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='color: lime;'>✅ No Brain Tumor Detected</h2>", unsafe_allow_html=True)

