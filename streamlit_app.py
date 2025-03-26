# Import statements
import streamlit as st
import tensorflow as tf
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import pandas as pd
from googleapiclient.http import MediaFileDownload
from datetime import datetime
import os

# Streamlit App
st.set_page_config(page_title="Butterfly Classification Model", layout="wide")

# Added a welcoming message with formatted text
welcome_text = st.markdown("<div style='text-align: left; padding: 20px;'><h1 style='color: #e8209c; display: inline-block;>Welcome</h1></div>", unsafe_allow_html=True)
subtitle_text = st.markdown("<div style='text-align: left; padding: 20px;'><h2 style='color: #4CAF50; display: inline-block;>to the</h2></div>", unsafe_allow_html=True)
butterfly_text = st.markdown("<div style='text-align: left; padding: 20px;'><h1 style='color: #e8209c; display: inline-block;>Butterfly Model</h1></div>", unsafe_allow_html=True)

def create_class_mapping_website(directory):
    """
    Create a mapping from class folder names to unique identifiers.

    Parameters:
    - directory (str): Path to the directory containing class subfolders.

    Returns:
    - dict: Mapping from class folder names to unique identifiers.
    """
    class_folders = sorted(os.listdir(directory))
    class_mapping = {folder: i for i, folder in enumerate(class_folders)}
    return class_mapping
    
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
        train_directory = "/content/train"
        class_mapping = create_class_mapping_website(train_directory)
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

# User option to choose the model
model_options = {
"ButterflyNet Model": "/content/ButterflyNet_Model.keras",
"EfficientNetB0 Model": "/content/custom_model_EfficientNetB0.keras",
"ResNet50 Model": "/content/resnet_custom_model.keras",
"VGG16 Model": "/content/vgg_custom_model.keras",
}
selected_model = st.selectbox("Select Model", list(model_options.keys()))
run_image_classification(model_options[selected_model])

