# Import statements
import streamlit as st
import tensorflow as tf
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import pandas as pd
from googleapiclient.http import MediaIoBaseDownload
from datetime import datetime
import os
import io

# Streamlit App
st.set_page_config(page_title="Butterfly Classification Model", layout="wide")

# Added a welcoming message with formatted text
welcome_text = st.markdown("<div style='text-align: left; padding: 20px;'><h1 style='color: #e8209c; display: inline-block;>Welcome</h1></div>", unsafe_allow_html=True)
subtitle_text = st.markdown("<div style='text-align: left; padding: 20px;'><h2 style='color: #4CAF50; display: inline-block;>to the</h2></div>", unsafe_allow_html=True)
butterfly_text = st.markdown("<div style='text-align: left; padding: 20px;'><h1 style='color: #e8209c; display: inline-block;>Butterfly Model</h1></div>", unsafe_allow_html=True)

#Dictionary of Butterfly Names
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

    
def run_image_classification(model_path):
    """
    Run image classification with a given model.

    Parameters:
        model_path (str): Path to the trained model file.

    Returns:
        None
    """
    # Load your trained model
    model = tf.keras.models.load_model(model_path)

# User option to choose the model
model_options = {
"Butterfly Model1": "https://drive.google.com/file/d/1t8QUiYFOWCMEoXMFWIfdf3T3DjUHBQJT/view?usp=sharing",
}
selected_model = st.selectbox("Select Model", list(model_options.keys()))
run_image_classification(model_options[selected_model])


# Function to preprocess the image
def preprocess_img(img_path):
  # Load image and check its size
  img = image.load_img(img_path)

  # Check if the image is already of size 224x224
  if img.size == (224, 224):
      # No changes needed, return the image as it is
      img_array = image.img_to_array(img)
      img_array = np.expand_dims(img_array, axis=0)
      img_array = preprocess_input(img_array)
      return img_array
  else:
      # Resize the image to 224x224 and perform preprocessing
      img = image.load_img(img_path, target_size=(224, 224))
      img_array = image.img_to_array(img)
      img_array = np.expand_dims(img_array, axis=0)
      img_array = preprocess_input(img_array)
      return img_array


# Streamlit App
st.markdown(
    """
    <div style='
        background-color: #f0f0f0;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        animation: fadeIn 1s ease-out;
    '>
        <h1 style='font-size:32px; color:#3366ff; text-align:center;'>Discover Butterfly or Moth Species</h1>
        <p style='font-size:18px; color:black; text-align:center;'>
            Upload your favorite butterfly/Moth picture and uncover its enchanting species!
        </p>
    </div>

    <style>
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Choose an image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image
    img_array = preprocess_img(uploaded_file)

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = tf.argmax(predictions, axis=1).numpy()
    confidence = predictions[0, predicted_class]

    # Load class mapping
    class_mapping = dic_butterfly 
    class_name = next((name for name, index in class_mapping.items() if index == predicted_class), f"Class {predicted_class[0]}")

    # Styling for the subheader
    subheader_style = """
        background-color: #4682b4;  /* SteelBlue color */
        color: #ffffff;  /* White color */
    """
    # Styling for the results container
    result_container_style = """
        background-color: #f8f8f8;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        animation: fadeIn 1s ease-out;
    """

    # Styling for the species name
    species_name_style = """
        font-size: 24px;
        color: #2e8b57;  /* SeaGreen color */
        margin-bottom: 10px;
    """

    # Styling for the confidence
    confidence_style = """
        font-size: 18px;
        color: #4169e1;  /* RoyalBlue color */
    """

    # Apply the styles
    st.markdown(
        f"""
        <div style='{result_container_style}'>
             <h2 style='{subheader_style}'>🦋 Your Butterfly/Moth Species Prediction 🌿</h2>
            <h3 style='{species_name_style}'>{class_name}</h3>
            <p style='{confidence_style}'>Confidence of Model: {confidence[0]:.2%}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

else:
    st.warning("Please upload an image to continue.")


