import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

# Load model and class names
model = tf.keras.models.load_model("skin_cancer_mobilenetv2.h5")
class_names = np.load("class_names.npy", allow_pickle=True)

# Full names
class_full_names = {
    "akiec": "Actinic Keratoses",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevus",
    "vasc": "Vascular Lesions"
}

# Descriptions
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

st.set_page_config(page_title="Skin Cancer Detection", layout="centered")

st.title("🩺 Skin Cancer Classification App")
st.write("Upload a skin lesion image to classify using AI")

st.warning(
    "⚠️ This application is for educational purposes only and is not a medical diagnosis."
)

uploaded_file = st.file_uploader(
    "Upload an image (jpg, png, jpeg)",
    type=["jpg", "png", "jpeg"]
)

def preprocess_image(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict"):
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
