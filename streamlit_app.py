import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileDownload
import os
import tempfile

def load_and_prepare_image(image_path, target_size):
    """Loads and preprocesses the image for prediction."""
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize the image to [0, 1]
    return image

def download_model_from_gdrive(file_id, destination):
    """Downloads a file from Google Drive given its file ID."""
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.http import MediaIoBaseDownload
        from googleapiclient.discovery import build
        import io
        from google.oauth2.credentials import Credentials

        # Define the scope
        SCOPES = ['https://www.googleapis.com/auth/drive']

        # Authenticate using the token.json file
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        service = build('drive', 'v3', credentials=creds)

        # Request the file
        request = service.files().get_media(fileId=file_id)
        fh = io.FileIO(destination, 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        return True
    except Exception as e:
        st.error(f"Error downloading the model: {e}")
        return False

def main():
    st.title("Keras Model Image Classifier")

    # Sidebar for model selection from Google Drive
    st.sidebar.header("Load Model from Google Drive")
    gdrive_file_id = st.sidebar.text_input("Enter Google Drive file ID for the model:")
    if st.sidebar.button("Download Model"):
        if gdrive_file_id:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp_file:
                model_path = tmp_file.name
                success = download_model_from_gdrive(gdrive_file_id, model_path)
                if success:
                    st.session_state["model_path"] = model_path
                    st.success("Model downloaded successfully!")

    # File browser for selecting an image
    st.sidebar.header("Select Image")
    uploaded_image = st.sidebar.file_uploader("Upload an image to classify:", type=["png", "jpg", "jpeg"])

    # Load the model if available
    if "model_path" in st.session_state:
        try:
            model = load_model(st.session_state["model_path"])
            st.success("Model loaded successfully!")

            # Display and classify the uploaded image
            if uploaded_image:
                # Display the selected image
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image", use_column_width=True)

                # Predict the class of the image
                if st.button("Classify Image"):
                    processed_image = load_and_prepare_image(uploaded_image, target_size=model.input_shape[1:3])
                    prediction = model.predict(processed_image)

                    # Assuming the model output is a probability distribution
                    predicted_class = np.argmax(prediction, axis=1)[0]
                    confidence = np.max(prediction) * 100

                    st.write(f"**Predicted Class:** {predicted_class}")
                    st.write(f"**Confidence:** {confidence:.2f}%")
        except Exception as e:
            st.error(f"Error loading model: {e}")
    else:
        st.info("Please download a model from Google Drive to get started.")

if __name__ == "__main__":
    main()
