import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Set Page Config
st.set_page_config(page_title="Pneumonia Detection AI", layout="centered")

@st.cache_resource
def load_pneumonia_model():
    # Load the model saved during training
    return tf.keras.models.load_model('pneumonia_model.h5')

model = load_pneumonia_model()

st.title("🫁 Chest X-Ray Diagnosis Tool")
st.write("Upload a Chest X-ray image to check for signs of Pneumonia.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded X-ray', use_column_width=True)
    
    # Preprocessing
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Prediction
    with st.spinner('Analyzing scan...'):
        prediction = model.predict(img_array)[0][0]
        prob_percent = prediction * 100

    # Display Results
    st.subheader("Results:")
    if prediction > 0.5:
        st.error(f"⚠️ High Risk: Pneumonia Detected")
        st.write(f"**Probability Score: {prob_percent:.2f}%**")
        st.warning("Note: This is an AI-assisted tool. Please consult a radiologist for final diagnosis.")
    else:
        st.success(f"✅ Low Risk: Normal Scan")
        st.write(f"**Probability Score: {prob_percent:.2f}%**")

st.markdown("---")
st.info("Model Info: Built using MobileNetV2 Transfer Learning on the Chest X-Ray (Pneumonia) Dataset.")
