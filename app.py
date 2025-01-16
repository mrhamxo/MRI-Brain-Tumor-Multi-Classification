import streamlit as st
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image
import os

# Load the trained model
model = load_model('models/model.h5')

# Class labels
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

# Title and description
st.title("MRI Tumor Detection System")
st.write("Upload an MRI image to detect if there is a tumor and its type.")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])

# Function to predict tumor type
def predict_tumor(image):
    IMAGE_SIZE = 128
    # Load the image in grayscale mode
    img = Image.open(image).convert('L').resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Convert to RGB by repeating the channel 3 times
    img_array = np.repeat(img_array, 3, axis=-1)

    # Make prediction using the model
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions, axis=1)[0]

    if class_labels[predicted_class_index] == 'notumor':
        return "No Tumor", confidence_score
    else:
        return f"Tumor: {class_labels[predicted_class_index]}", confidence_score

# If file is uploaded, process it
if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded MRI Image", use_column_width=True)
    
    # Predict tumor and display result
    result, confidence = predict_tumor(uploaded_file)
    st.write(f"**Result:** {result}")
    st.write(f"**Confidence:** {confidence * 100:.2f}%")
