import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from PIL import Image

def load_and_prepare_image(image_path, target_size):
    """Loads and preprocesses the image for prediction."""
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize the image to [0, 1]
    return image

def main():
    st.title("Keras Model Image Classifier")

    # Sidebar for model upload
    st.sidebar.header("Upload Model")
    model_file = st.sidebar.file_uploader("Upload your Keras model (.keras file):", type=["keras"])

    # Sidebar for image folder selection
    st.sidebar.header("Select Image Folder")
    folder_path = st.sidebar.text_input("Enter the path to the folder containing images:")

    if model_file:
        try:
            # Load the model
            model = load_model(model_file)
            st.success("Model loaded successfully!")

            # Check if folder path is valid
            if folder_path and os.path.isdir(folder_path):
                image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(("png", "jpg", "jpeg"))]

                if image_files:
                    selected_image = st.selectbox("Select an image to classify:", image_files)

                    if selected_image:
                        # Display the selected image
                        image_path = os.path.join(folder_path, selected_image)
                        image = Image.open(image_path)
                        st.image(image, caption=f"Selected Image: {selected_image}", use_column_width=True)

                        # Predict the class of the image
                        if st.button("Classify Image"):
                            processed_image = load_and_prepare_image(image_path, target_size=model.input_shape[1:3])
                            prediction = model.predict(processed_image)

                            # Assuming the model output is a probability distribution
                            predicted_class = np.argmax(prediction, axis=1)[0]
                            confidence = np.max(prediction) * 100

                            st.write(f"**Predicted Class:** {predicted_class}")
                            st.write(f"**Confidence:** {confidence:.2f}%")
                else:
                    st.warning("No images found in the specified folder.")
            else:
                st.warning("Please enter a valid folder path containing images.")

        except Exception as e:
            st.error(f"Error loading model: {e}")
    else:
        st.info("Please upload a Keras model to get started.")

if __name__ == "__main__":
    main()
