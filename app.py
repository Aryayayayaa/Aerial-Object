import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import base64
from tensorflow.keras.models import load_model # Explicit import for clarity

# --- Configuration ---
MODEL_PATH = 'best_Optimized_MobileNetV2.h5'
BACKGROUND_IMAGE_PATH = 'bg.jpg'
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ['Bird (Harmless)', 'UAV/Drone (Threat)'] # Assuming 0: Bird, 1: Drone

# --- Utility Functions ---

def set_background(image_file):
    """
    Sets the Streamlit app's background image using CSS injection.
    Requires the image file (bg.jpg) to be in the same directory.
    """
    try:
        with open(image_file, "rb") as f:
            data = f.read()
        
        # Encode the image data to base64
        bin_str = base64.b64encode(data).decode()
        
        # CSS to inject the background image
        page_bg_img = f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{bin_str}");
            background-size: cover;
            background-attachment: fixed;
            background-repeat: no-repeat;
            background-position: center center;
        }}
        /* Custom styling for better readability over the image */
        h1, h2, h3, h4, .stText {{
            color: #ffffff;
            text-shadow: 2px 2px 4px #000000;
        }}
        .css-1d391kg, .stFileUploader {{ 
            background-color: rgba(30, 30, 30, 0.7); /* Dark semi-transparent background for file uploader/sidebar */
            padding: 10px;
            border-radius: 10px;
        }}
        .stMarkdown, .stText {{
            color: #ffffff !important;
        }}
        </style>
        """
        st.markdown(page_bg_img, unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"Background image '{image_file}' not found. Using default Streamlit theme.")
    except Exception as e:
        st.error(f"Error loading background image: {e}")

@st.cache_resource
def load_classification_model(path):
    """
    Loads the Keras model and caches it to prevent reloading on every rerun.
    """
    if not os.path.exists(path):
        st.error(f"Model file not found at: {path}. Please place '{path}' in the same directory.")
        return None
    try:
        model = load_model(path)
        return model
    except Exception as e:
        st.error(f"Failed to load model from {path}: {e}")
        return None

def preprocess_image(image):
    """
    Preprocesses the PIL image for MobileNetV2 inference (Resize, Normalization, Batch dimension).
    """
    # Resize image to the required size (224x224 for MobileNetV2)
    image = image.resize(IMAGE_SIZE)
    # Convert image to numpy array
    image_array = np.array(image)
    # Scale pixel values to [0, 1]
    image_array = image_array / 255.0
    # Add batch dimension (1, 224, 224, 3)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict(model, processed_image):
    """
    Performs inference and returns the class index and confidence.
    """
    predictions = model.predict(processed_image)
    # Assuming a binary classification (single output neuron with sigmoid, or two with softmax)
    
    # Check if the output is a single value (sigmoid) or two values (softmax)
    if predictions.shape[-1] == 1:
        # Sigmoid output (0 to 1)
        confidence = predictions[0][0]
        # Class 1 (Drone) is > 0.5, Class 0 (Bird) is <= 0.5
        predicted_class_index = 1 if confidence > 0.5 else 0
        
        # Normalize confidence to the predicted class
        if predicted_class_index == 0:
            confidence = 1.0 - confidence # Confidence for Bird
        
    elif predictions.shape[-1] == len(CLASS_NAMES):
        # Softmax output
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        confidence = predictions[0][predicted_class_index]
    else:
        st.error(f"Unexpected model output shape: {predictions.shape}. Expected (1, 1) or (1, {len(CLASS_NAMES)})")
        return None, None
        
    return predicted_class_index, confidence

# --- Main Streamlit App ---

def main():
    # 1. Set the custom background image
    set_background(BACKGROUND_IMAGE_PATH)

    st.title("🦅 UAV/Drone vs. Bird Classifier")
    st.markdown("---")
    
    st.header("Upload Image for Aerial Threat Assessment")

    # 2. Load the model
    model = load_classification_model(MODEL_PATH)

    if model is None:
        # If model loading failed, stop execution
        st.stop()

    # 3. Image Upload UI
    uploaded_file = st.file_uploader(
        "Choose an image of an aerial object (PNG, JPG, JPEG)", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        try:
            # Load and display the image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image', use_column_width=True)

            with st.spinner('Analyzing image...'):
                # 4. Preprocess and Predict
                processed_image = preprocess_image(image)
                predicted_class_index, confidence = predict(model, processed_image)

            st.markdown("---")
            st.subheader("Classification Result")

            if predicted_class_index is not None:
                predicted_class = CLASS_NAMES[predicted_class_index]
                confidence_percent = confidence * 100

                # 5. Display Prediction and Confidence
                if predicted_class_index == 1:
                    # Drone detected - use caution/danger styling
                    st.error(f"## 🚨 **PREDICTION: {predicted_class}**")
                    st.markdown(f"**Confidence:** **<span style='color:red;'>{confidence_percent:.2f}%</span>**", unsafe_allow_html=True)
                    st.warning("⚠️ High-priority security alert: This object is classified as a potential UAV/Drone threat.")
                else:
                    # Bird detected - use success/safe styling
                    st.success(f"## ✅ **PREDICTION: {predicted_class}**")
                    st.markdown(f"**Confidence:** <span style='color:lightgreen;'>{confidence_percent:.2f}%</span>", unsafe_allow_html=True)
                    st.info("ℹ️ Object confirmed as a common bird. Low security risk.")

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
            st.exception(e)


if __name__ == '__main__':
    # Set page config for a wider layout
    st.set_page_config(
        page_title="UAV/Bird Classifier", 
        layout="wide", 
        initial_sidebar_state="expanded"
    )
    main()