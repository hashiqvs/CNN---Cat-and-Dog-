import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐾",
    layout="centered",
)

st.title("🐱 Cat vs 🐶 Dog Classifier")
st.markdown("Upload an image and the CNN model will predict whether it's a **cat** or a **dog**.")

# ── Model loading ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the saved Keras model (supports both .h5 and .keras formats)."""
    for model_path in ["cats_dogs_cnn.h5", "cats_dogs_cnn.keras"]:
        if os.path.exists(model_path):
            return tf.keras.models.load_model(model_path), model_path
    return None, None

model, model_path = load_model()

if model is None:
    st.error(
        "⚠️ Model file not found.\n\n"
        "Please place `cats_dogs_cnn.h5` (or `cats_dogs_cnn.keras`) "
        "in the same directory as this app, then restart."
    )
    st.stop()
else:
    st.success(f"✅ Model loaded from `{model_path}`")

# ── Image upload ────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    # Display the image
    img_pil = Image.open(uploaded_file).convert("RGB")
    st.image(img_pil, caption="Uploaded Image", use_container_width=True)

    # ── Preprocess ──────────────────────────────────────────────────────────────
    img_resized = img_pil.resize((64, 64))
    img_array  = np.array(img_resized, dtype=np.float32) / 255.0
    img_batch  = np.expand_dims(img_array, axis=0)        # shape: (1, 64, 64, 3)

    # ── Predict ─────────────────────────────────────────────────────────────────
    with st.spinner("Classifying…"):
        prediction = model.predict(img_batch, verbose=0)[0][0]

    label      = "Dog 🐶" if prediction > 0.5 else "Cat 🐱"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    # ── Results ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    col1, col2 = st.columns(2)
    col1.metric("Prediction", label)
    col2.metric("Confidence", f"{confidence * 100:.1f}%")

    # Confidence bar
    st.progress(float(confidence))
    st.caption(
        f"Raw sigmoid output: `{prediction:.4f}` "
        f"(threshold = 0.5 → ≤ 0.5 = Cat, > 0.5 = Dog)"
    )

# ── Sidebar: model info ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ Model Info")
    st.markdown(
        """
        **Architecture:** Sequential CNN  
        **Input size:** 64 × 64 × 3 (RGB)  
        **Layers:**
        - Conv2D (32 filters) + MaxPool  
        - Conv2D (32 filters) + MaxPool  
        - Flatten  
        - Dense (128, ReLU) + Dropout (0.5)  
        - Dense (1, Sigmoid)  

        **Training:** 10 epochs, Adam optimizer  
        **Val accuracy:** ~76%
        """
    )
    st.markdown("---")
    st.markdown("Built with **TensorFlow / Keras** + **Streamlit**")