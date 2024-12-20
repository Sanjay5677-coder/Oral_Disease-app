import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('oral_disease_recognition_model.h5')
    return model

model = load_model()

# Class names (modify based on your dataset)
CLASS_NAMES = ['dental_caries', 'gingivitis', 'healthy']

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to match model's input shape
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit app
st.title("Oral Disease Classification")
st.write("Upload an image to classify oral diseases.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    st.write("Classifying...")
    input_image = preprocess_image(image)
    
    # Make prediction
    predictions = model.predict(input_image)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    
    # Display the result
    st.write(f"Predicted Class: **{predicted_class}**")
    st.write(f"Confidence: **{confidence:.2f}%**")
