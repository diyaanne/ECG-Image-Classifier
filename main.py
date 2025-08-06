import tensorflow as tf
from tensorflow import keras
import joblib
import streamlit as st
from tensorflow.keras.models import Model
import numpy as np
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import io

# Constants
IMG_SIZE = (224, 224)
labels = ["History of MI", "Myocardial Infraction Patients", "Normal Person", "abnormal heartbeat"]

# Load MobileNetV2 base model
base_model = keras.applications.MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
    pooling='avg'
)

# Freeze the base model and create a feature extractor model
model_frozen = Model(inputs=base_model.input, outputs=base_model.output)

# Load the trained classifier (assumes it's compatible with extracted features)
model = joblib.load('MLP_best_model.joblib')

def convert_tf_dataset(img_array, model_frozen):
    """
    Accepts a numpy array image, resizes and preprocesses it,
    and returns feature vector using MobileNetV2.
    """
    # Resize image
    img_resized = cv2.resize(img_array, IMG_SIZE)

    # Ensure 3 channels
    if len(img_resized.shape) == 2:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    elif img_resized.shape[2] == 1:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)

    # Expand dimensions to match model input
    img_batch = np.expand_dims(img_resized, axis=0)

    # Preprocess image
    img_preprocessed = preprocess_input(img_batch)

    # Extract features
    features = model_frozen.predict(img_preprocessed, verbose=False)
    return features

# Streamlit UI
st.title("Cardiac Condition Classifier")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image bytes
    image = Image.open(uploaded_file).convert('RGB')
    img_np = np.array(image)

    # Convert to grayscale if needed for thresholding
    gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Crop the image (ensure valid indices)
    cropped = gray_img[300:1500, :] if gray_img.shape[0] > 1500 else gray_img

    # Threshold the image
    _, binary = cv2.threshold(cropped, 50, 255, cv2.THRESH_BINARY)

    # Convert back to 3 channels for MobileNet
    binary_3ch = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

    # Feature extraction and prediction
    features = convert_tf_dataset(binary_3ch, model_frozen)
    prediction = model.predict(features)

    # Output prediction
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Prediction:", labels[int(prediction[0])])
    st.write("Prediction Index:", int(prediction[0]))
