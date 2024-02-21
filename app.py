import streamlit as st
import tensorflow as tf
import numpy as np
# from tensorflow.keras.models import load_model
import urllib.request
from PIL import Image
import requests
from io import BytesIO
import os

st.set_page_config(
    page_title="SynthCheck",
    page_icon="🤖") #layout='wide'

st.title('SynthCheck: A Synthetic Image Identifier ')

# GitHub URL of your model file (replace with your actual URL)
github_model_url = "https://raw.githubusercontent.com/ShreyashSomvanshi/test_synthcheck/main/firstModel.h5"


# Download the model file
response = requests.get(github_model_url)
with open("firstModel.h5", "wb") as f:
    f.write(response.content)

# Load the model using TensorFlow
model = tf.keras.models.load_model("firstModel.h5")


def classify_image(file_path):
    # model = load_model('firstModel.h5')
    image = Image.open(file_path) # reading the image
    image = image.resize((32, 32)) # resizing the image to fit the trained model   
    img = np.asarray(image) # converting it to numpy array
    img = np.expand_dims(img/255, 0)
    predictions = model.predict(img) # predicting the label
    if predictions > 0.5:
        res = 'Predicted class: REAL'
    else:
        res = 'Predicted class: SYNTHETIC'
    return res

    
    
st.write("Upload an image to check whether it is a fake or real image.")

file_uploaded = st.file_uploader("Choose the Image File", type=["jpg", "png", "jpeg"])
if file_uploaded is not None:
    res = classify_image(file_uploaded)
    c1, buff, c2 = st.columns([2, 0.5, 2])
    c1.image(file_uploaded, use_column_width=True)
    c2.subheader("Classification Result")
    c2.write("The image is classified as **{}**.".format(res.title()))

st.button('Check', use_container_width=True) #use_container_width=True
