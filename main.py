import tensorflow as tf
from tensorflow import keras
import joblib
import streamlit as st 
from tensorflow.keras.models import Model
import sklearn
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
IMG_SIZE = (224, 224)


# Download the model, valid alpha values [0.25,0.35,0.5,0.75,1]
base_model = keras.applications.MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
    pooling='avg'
)


# base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
# Add average pooling to the base
x = base_model.output
model_frozen = Model(inputs=base_model.input,outputs=x)

# Load the model from a .pkl file
model = joblib.load('MLP_best_model.pkl')

def convert_tf_dataset(img_path, model):
    # This function passes all images provided in PATH
    # and passes them through the model.
    # The result is a featurized image along with labels

    IMG_SIZE = (224, 224)
  
      
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    data = model.predict(img_preprocessed, verbose=False)

    return data

uploaded_file = st.file_uploader("upload an image", type = ["jpg", "jpeg","png" ]) 
if uploaded_file is not None:
    features = convert_tf_dataset(uploaded_file, model_frozen)
    prediction = model.predict(features)
    st.write("prediction: ", prediction)

