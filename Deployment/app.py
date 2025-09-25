import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the model
model = tf.keras.models.load_model("/content/best_model.h5")

# Get input shape from model (ignore batch dimension)
input_shape = model.input_shape[1:]  # e.g. (224, 224, 1) or (224, 224, 3)
img_height, img_width, img_channels = input_shape

# Class names
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Streamlit page config
st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="centered")

# Add a banner image from URL
st.image(
    "https://png.pngtree.com/thumb_back/fh260/background/20250417/pngtree-glowing-human-brain-image_17208331.jpg",
    caption="Brain MRI Example",
    use_column_width=True
)

st.title("🧠 Brain Tumor Detection App")
st.write("Upload an MRI scan and the model will predict the tumor type.")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file)

    # Force correct channels
    if img_channels == 1:
        image = image.convert("L")  # grayscale
    else:
        image = image.convert("RGB")  # RGB

    # Display uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((img_width, img_height))
    img_array = np.array(img) / 255.0

    # Add missing dimensions if grayscale
    if img_channels == 1:
        img_array = np.expand_dims(img_array, axis=-1)  # (H,W,1)

    img_array = np.expand_dims(img_array, axis=0)  # (1,H,W,C)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)

    # Result
    result = f"🔍 Predicted: **{class_names[predicted_class]}** ({confidence*100:.2f}% confidence)"
    st.subheader("Result:")
    st.success(result)
