import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Skin Cancer Detection",
    layout="centered",
    page_icon="🩺"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("v2.h5")

model = load_model()

# ---------------- CLASS NAMES ----------------
class_names = np.array([
    "akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"
])

class_full_names = {
    "akiec": "Actinic Keratoses",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevus",
    "vasc": "Vascular Lesions"
}

descriptions = {
    "akiec": "Pre-cancerous skin lesion caused by sun damage.",
    "bcc": "Common skin cancer that grows slowly and rarely spreads.",
    "bkl": "Benign skin growth, not cancerous.",
    "df": "Benign fibrous skin tumor.",
    "mel": "Dangerous form of skin cancer that can spread quickly.",
    "nv": "Benign moles that are usually harmless.",
    "vasc": "Skin lesions related to blood vessels."
}

high_risk = ["mel", "bcc", "akiec"]

# ---------------- UI ----------------
st.title("🩺 Skin Cancer Classification App")
st.write("Upload a skin lesion image to classify using AI")

st.warning(
    "⚠️ This application is for educational purposes only and "
    "is NOT a medical diagnosis."
)

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload an image (JPG, JPEG, PNG, JFIF)",
    type=["jpg", "jpeg", "png", "jfif"]
)

# ---------------- PREPROCESS ----------------
def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ---------------- PREDICTION ----------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing image..."):
            processed_img = preprocess_image(image)
            prediction = model.predict(processed_img)

            class_index = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            predicted_class = class_names[class_index]

        st.subheader("🧠 Prediction Result")
        st.success(f"**{class_full_names[predicted_class]}**")
        st.write(f"**Confidence:** {confidence:.2f}%")

        if predicted_class in high_risk:
            st.error("⚠️ High Risk Lesion — Consult a dermatologist")
        else:
            st.success("✅ Low Risk Lesion")

        st.subheader("📘 Description")
        st.write(descriptions[predicted_class])

        st.subheader("📊 Probability Distribution")
        df_probs = pd.DataFrame({
            "Class": [class_full_names[c] for c in class_names],
            "Probability": prediction[0]
        })
        st.bar_chart(df_probs.set_index("Class"))
