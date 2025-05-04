import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import time
import pandas as pd
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from googletrans import Translator, LANGUAGES  # Updated import

# ✅ Initialize session state
if "user_points" not in st.session_state:
    st.session_state.user_points = 0

# ✅ Must be the first Streamlit command
st.set_page_config(
    page_title="Ship & Oil Spill Detection",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================
# 🎨 CUSTOM THEME SETUP
# ======================
def apply_theme(theme):
    if theme == "dark":
        st.markdown(
            """
            <style>
            /* General Styles */
            body {
                background-color: #0E1117;
                color: #FFFFFF;
            }
            .stApp {
                background: #0E1117;
            }
            /* Centered Title */
            .centered-title {
                text-align: center;
                font-size: 2.5em;
                font-weight: bold;
                color: #4CAF50;
                margin-bottom: 20px;
            }
            /* Button Styles */
            .stButton button {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 16px;
                font-weight: bold;
                transition: all 0.3s ease;
            }
            .stButton button:hover {
                background-color: #45a049;
                transform: scale(1.05);
            }
            /* File Uploader Styles */
            .stFileUploader div {
                border: 2px dashed #4CAF50;
                border-radius: 8px;
                padding: 20px;
                text-align: center;
                background-color: rgba(76, 175, 80, 0.1);
            }
            /* Result Box Styles */
            .stSuccess {
                background-color: #4CAF50;
                color: white;
                border-radius: 8px;
                padding: 15px;
                text-align: center;
            }
            .stWarning {
                background-color: #FFA000;
                color: white;
                border-radius: 8px;
                padding: 15px;
                text-align: center;
            }
            .stInfo {
                background-color: #2979FF;
                color: white;
                border-radius: 8px;
                padding: 15px;
                text-align: center;
            }
            /* Sidebar Styles */
            .css-1d391kg {
                background-color: #1E1E2F;
                color: #FFFFFF;
            }
            .stSidebar .stMarkdown {
                color: #FFFFFF;
            }
            .stSidebar .stRadio label {
                color: #FFFFFF;
            }
            .stSidebar .stFileUploader label {
                color: #FFFFFF;
            }
            /* Footer Styles */
            .footer {
                color: #FFFFFF;
                text-align: center;
                font-size: 0.9em;
                margin-top: 20px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            /* General Styles */
            body {
                background-color: #FFFFFF;
                color: #000000;
            }
            .stApp {
                background: #FFFFFF;
            }
            /* Centered Title */
            .centered-title {
                text-align: center;
                font-size: 2.5em;
                font-weight: bold;
                color: #4CAF50;
                margin-bottom: 20px;
            }
            /* Button Styles */
            .stButton button {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 16px;
                font-weight: bold;
                transition: all 0.3s ease;
            }
            .stButton button:hover {
                background-color: #45a049;
                transform: scale(1.05);
            }
            /* File Uploader Styles */
            .stFileUploader div {
                border: 2px dashed #4CAF50;
                border-radius: 8px;
                padding: 20px;
                text-align: center;
                background-color: rgba(76, 175, 80, 0.1);
            }
            /* Result Box Styles */
            .stSuccess {
                background-color: #4CAF50;
                color: white;
                border-radius: 8px;
                padding: 15px;
                text-align: center;
            }
            .stWarning {
                background-color: #FFA000;
                color: white;
                border-radius: 8px;
                padding: 15px;
                text-align: center;
            }
            .stInfo {
                background-color: #2979FF;
                color: white;
                border-radius: 8px;
                padding: 15px;
                text-align: center;
            }
            /* Sidebar Styles */
            .css-1d391kg {
                background-color: #F0F0F0;
                color: #000000;
            }
            .stSidebar .stMarkdown {
                color: #000000;
            }
            .stSidebar .stRadio label {
                color: #000000;
            }
            .stSidebar .stFileUploader label {
                color: #000000;
            }
            /* Footer Styles */
            .footer {
                color: #000000;
                text-align: center;
                font-size: 0.9em;
                margin-top: 20px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

# ======================
# 🔧 CORE FUNCTIONS
# ======================
@st.cache_resource
def load_model(model_path):
    """Load TensorFlow model with error handling."""
    # if not os.path.exists(model_path):
    #     st.error(f"🚨 Model not found: {model_path}")
    #     return None
    # try:
    #     model = tf.keras.models.load_model(model_path)
    #     st.success(f"✅ Model loaded: {model_path}")
    #     return model
    # except Exception as e:
    #     st.error(f"🚨 Model loading failed: {e}")
    #     return None

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model prediction."""
    try:
        image = image.resize(target_size)
        image = np.array(image) / 255.0  # Normalize
        return np.expand_dims(image, axis=0)  # Add batch dimension
    except Exception as e:
        st.error(f"🚨 Image processing error: {e}")
        return None

def predict(image, model, label_dict, confidence_threshold=0.5):
    """Run prediction with confidence thresholding."""
    if model is None:
        return "Model Not Loaded", 0.0
    
    processed_image = preprocess_image(image)
    if processed_image is None:
        return "Image Processing Failed", 0.0
    
    try:
        prediction = model.predict(processed_image)
        class_index = np.argmax(prediction)
        confidence = float(prediction[0][class_index])
        
        if confidence >= confidence_threshold:
            return label_dict[class_index], confidence
        else:
            return "Low Confidence Prediction", confidence
    except Exception as e:
        st.error(f"🚨 Prediction error: {e}")
        return "Prediction Failed", 0.0

def draw_boxes(image, boxes):
    """Draw bounding boxes on image."""
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image

def translate_text(text, dest_language="en"):
    """Translate text with fallback."""
    try:
        return translator.translate(text, dest=dest_language).text
    except:
        return text  # Fallback to original text

def award_points(points):
    """Award points with visual feedback."""
    st.session_state.user_points += points
    st.balloons()
    st.success(f"🎉 +{points} points!")

# ======================
# 🎥 REAL-TIME CAMERA CLASS
# ======================
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.detection_type = st.session_state.get("detection_type", "Ship Detection")
        self.model = load_model(MODEL_PATHS[self.detection_type])

    def transform(self, frame):
        try:
            image = Image.fromarray(frame.to_ndarray(format="bgr24"))
            label_dict = LABEL_DICT[self.detection_type]
            prediction, confidence = predict(image, self.model, label_dict)
            
            # Draw boxes if ship detected
            if prediction == "Ship Detected":
                boxes = [(50, 50, 100, 100)]  # Example (replace with real detection)
                frame = draw_boxes(frame.to_ndarray(format="bgr24"), boxes)
            
            return frame
        except Exception as e:
            st.error(f"Camera error: {e}")
            return frame

# ======================
# 📊 CONSTANTS & CONFIG
# ======================
MODEL_PATHS = {
    "Ship Detection": "ship_detection.h5",
    "Oil Spill Detection": "oil_spill.h5",
}

LABEL_DICT = {
    "Ship Detection": {0: "No Ship", 1: "Ship Detected"},
    "Oil Spill Detection": {0: "No Oil Spill", 1: "Oil Spill Detected"},
}

LANGUAGE_MAP = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
}

# ======================
# 🖥️ STREAMLIT UI
# ======================
# Theme Toggle
theme = st.sidebar.radio("Theme", ["Light Mode", "Dark Mode"])
apply_theme("dark" if theme == "Dark Mode" else "light")

# Sidebar Controls
st.sidebar.header("Settings")
detection_type = st.sidebar.radio("Detection Type", list(MODEL_PATHS.keys()))
st.session_state.detection_type = detection_type  # For camera class

confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)
language = st.sidebar.selectbox("Language", list(LANGUAGE_MAP.keys()))
offline_mode = st.sidebar.checkbox("Offline Mode")

# Load Model
model = load_model(MODEL_PATHS[detection_type])

# Main Title
title_text = f"🔍 {detection_type} using AI"
if language != "English" and not offline_mode:
    title_text = translate_text(title_text, LANGUAGE_MAP[language])
st.markdown(f'<h1 class="centered-title">{title_text}</h1>', unsafe_allow_html=True)

# File Uploader
uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# ======================
# 🖼️ IMAGE PROCESSING
# ======================
if uploaded_file:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Detect", key="detect_btn"):
            with st.spinner("Analyzing..."):
                time.sleep(1)  # Simulate processing
                prediction, confidence = predict(image, model, LABEL_DICT[detection_type], confidence_threshold)

                # Display Results
                st.subheader("🔍 Results")
                col1, col2 = st.columns(2)
                col1.metric("Prediction", prediction)
                col2.metric("Confidence", f"{confidence*100:.2f}%")

                # Visual Feedback
                if "Detected" in prediction:
                    if "Ship" in prediction:
                        award_points(10)
                        st.markdown('<div class="stSuccess">✅ Ship detected!</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="stWarning">⚠️ Oil spill detected!</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="stInfo">🔍 No detection</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"❌ Error: {e}")

# ======================
# 🎥 CAMERA FEED
# ======================
if st.sidebar.checkbox("Enable Camera"):
    # st.warning("Real-time detection may reduce performance")
    webrtc_streamer(key="camera_feed", video_transformer_factory=VideoTransformer)

# ======================
# 📜 FOOTER
# ======================
st.sidebar.markdown("---")
st.sidebar.markdown(
    f'<div class="footer">👨‍💻 Points: {st.session_state.user_points} | Made with Streamlit</div>',
    unsafe_allow_html=True
)