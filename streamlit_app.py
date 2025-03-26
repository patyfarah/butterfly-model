import streamlit as st
import tensorflow as tf
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import pandas as pd
import gdown
import os
from io import BytesIO

# Streamlit App
st.set_page_config(page_title="Butterfly Classification Model", layout="centered")

# Welcome message
st.markdown("<h1 style='color: #e8209c;'>Welcome to the Butterfly Model</h1>", unsafe_allow_html=True)

# Butterfly dictionary
dic_butterfly = {
 np.str_('ADONIS'): 0,
 np.str_('AFRICAN GIANT SWALLOWTAIL'): 1,
 np.str_('AMERICAN SNOOT'): 2,
 np.str_('AN 88'): 3,
 np.str_('APPOLLO'): 4,
 np.str_('ARCIGERA FLOWER MOTH'): 5,
 np.str_('ATALA'): 6,
 np.str_('ATLAS MOTH'): 7,
 np.str_('BANDED ORANGE HELICONIAN'): 8,
 np.str_('BANDED PEACOCK'): 9,
 np.str_('BANDED TIGER MOTH'): 10,
 np.str_('BECKERS WHITE'): 11,
 np.str_('BIRD CHERRY ERMINE MOTH'): 12,
 np.str_('BLACK HAIRSTREAK'): 13,
 np.str_('BLUE MORPHO'): 14,
 np.str_('BLUE SPOTTED CROW'): 15,
 np.str_('BROOKES BIRDWING'): 16,
 np.str_('BROWN ARGUS'): 17,
 np.str_('BROWN SIPROETA'): 18,
 np.str_('CABBAGE WHITE'): 19,
 np.str_('CAIRNS BIRDWING'): 20,
 np.str_('CHALK HILL BLUE'): 21,
 np.str_('CHECQUERED SKIPPER'): 22,
 np.str_('CHESTNUT'): 23,
 np.str_('CINNABAR MOTH'): 24,
 np.str_('CLEARWING MOTH'): 25,
 np.str_('CLEOPATRA'): 26,
 np.str_('CLODIUS PARNASSIAN'): 27,
 np.str_('CLOUDED SULPHUR'): 28,
 np.str_('COMET MOTH'): 29,
 np.str_('COMMON BANDED AWL'): 30,
 np.str_('COMMON WOOD-NYMPH'): 31,
 np.str_('COPPER TAIL'): 32,
 np.str_('CRECENT'): 33,
 np.str_('CRIMSON PATCH'): 34,
 np.str_('DANAID EGGFLY'): 35,
 np.str_('EASTERN COMA'): 36,
 np.str_('EASTERN DAPPLE WHITE'): 37,
 np.str_('EASTERN PINE ELFIN'): 38,
 np.str_('ELBOWED PIERROT'): 39,
 np.str_('EMPEROR GUM MOTH'): 40,
 np.str_('GARDEN TIGER MOTH'): 41,
 np.str_('GIANT LEOPARD MOTH'): 42,
 np.str_('GLITTERING SAPPHIRE'): 43,
 np.str_('GOLD BANDED'): 44,
 np.str_('GREAT EGGFLY'): 45,
 np.str_('GREAT JAY'): 46,
 np.str_('GREEN CELLED CATTLEHEART'): 47,
 np.str_('GREEN HAIRSTREAK'): 48,
 np.str_('GREY HAIRSTREAK'): 49,
 np.str_('HERCULES MOTH'): 50,
 np.str_('HUMMING BIRD HAWK MOTH'): 51,
 np.str_('INDRA SWALLOW'): 52,
 np.str_('IO MOTH'): 53,
 np.str_('Iphiclus sister'): 54,
 np.str_('JULIA'): 55,
 np.str_('LARGE MARBLE'): 56,
 np.str_('LUNA MOTH'): 57,
 np.str_('MADAGASCAN SUNSET MOTH'): 58,
 np.str_('MALACHITE'): 59,
 np.str_('MANGROVE SKIPPER'): 60,
 np.str_('MESTRA'): 61,
 np.str_('METALMARK'): 62,
 np.str_('MILBERTS TORTOISESHELL'): 63,
 np.str_('MONARCH'): 64,
 np.str_('MOURNING CLOAK'): 65,
 np.str_('OLEANDER HAWK MOTH'): 66,
 np.str_('ORANGE OAKLEAF'): 67,
 np.str_('ORANGE TIP'): 68,
 np.str_('ORCHARD SWALLOW'): 69,
 np.str_('PAINTED LADY'): 70,
 np.str_('PAPER KITE'): 71,
 np.str_('PEACOCK'): 72,
 np.str_('PINE WHITE'): 73,
 np.str_('PIPEVINE SWALLOW'): 74,
 np.str_('POLYPHEMUS MOTH'): 75,
 np.str_('POPINJAY'): 76,
 np.str_('PURPLE HAIRSTREAK'): 77,
 np.str_('PURPLISH COPPER'): 78,
 np.str_('QUESTION MARK'): 79,
 np.str_('RED ADMIRAL'): 80,
 np.str_('RED CRACKER'): 81,
 np.str_('RED POSTMAN'): 82,
 np.str_('RED SPOTTED PURPLE'): 83,
 np.str_('ROSY MAPLE MOTH'): 84,
 np.str_('SCARCE SWALLOW'): 85,
 np.str_('SILVER SPOT SKIPPER'): 86,
 np.str_('SIXSPOT BURNET MOTH'): 87,
 np.str_('SLEEPY ORANGE'): 88,
 np.str_('SOOTYWING'): 89,
 np.str_('SOUTHERN DOGFACE'): 90,
 np.str_('STRAITED QUEEN'): 91,
 np.str_('TROPICAL LEAFWING'): 92,
 np.str_('TWO BARRED FLASHER'): 93,
 np.str_('ULYSES'): 94,
 np.str_('VICEROY'): 95,
 np.str_('WHITE LINED SPHINX MOTH'): 96,
 np.str_('WOOD SATYR'): 97,
 np.str_('YELLOW SWALLOW TAIL'): 98,
 np.str_('ZEBRA LONG WING'): 99}

# Function to download and load the model
#@st.cache
def load_model_from_url(model_url):
    # Download the model using gdown (Make sure the URL is a direct Google Drive link)
    output_path = "/tmp/butterfly_model.keras"
    gdown.download(model_url, output_path, quiet=False)
    model = tf.keras.models.load_model(output_path)
    return model

# Function to preprocess the image
def preprocess_img(uploaded_file):
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Available model options (make sure the links are direct downloadable URLs)
model_options = {
    "Butterfly Model1": "https://drive.google.com/uc?export=download&id=1t8QUiYFOWCMEoXMFWIfdf3T3DjUHBQJT",  # Update this with the actual direct download link
    "Butterfly VGG16":"https://drive.google.com/uc?export=download&id=1Mz8yeVckw85VEIqgrsaLDVivyihtp7hW"
}

# Model selection
selected_model = st.selectbox("Select Model", list(model_options.keys()))
model_url = model_options[selected_model]

# Load model once selected
model = load_model_from_url(model_url)

# Streamlit user interface for image upload
st.markdown(
    """
    <div style='background-color: #f0f0f0; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);'>
        <h1 style='font-size:32px; color:#3366ff; text-align:center;'>Discover Butterfly or Moth Species</h1>
        <p style='font-size:18px; color:black; text-align:center;'>Upload your favorite butterfly/Moth picture for classification!</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)
    
    # Preprocess the image
    img_array = preprocess_img(uploaded_file)

    # Predict the class
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]  # Convert to a scalar value
    confidence = predictions[0, predicted_class]
    
    # Get butterfly name from dictionary
    class_mapping = dic_butterfly
    class_name = next((name for name, index in class_mapping.items() if index == predicted_class), f"Class {predicted_class}")


    # Styling the output
    st.markdown(
        f"""
        <div style="background-color: #f8f8f8; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
            <h2 style="background-color: #4682b4; color: white; padding: 10px;">🦋 Butterfly/Moth Species Prediction 🌿</h2>
            <h3 style="font-size: 24px; color: #2e8b57;">{class_name} {predicted_class}</h3>
            <p style="font-size: 18px; color: #4169e1;">Confidence: {confidence * 100:.2f}%</p>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.warning("Please upload an image to continue.")
