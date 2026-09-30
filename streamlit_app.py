import os
import sys

# 1. Guarantee working directory is SimpleHTR/src
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
os.chdir(SRC_DIR)

import streamlit as st
import numpy as np
import cv2
from PIL import Image

# 2. Render the page UI immediately
st.set_page_config(
    page_title="Handwriting Recognition (SimpleHTR)",
    page_icon="✍️",
    layout="centered"
)

st.title("✍️ Handwriting OCR Web Interface")
st.markdown("Upload a single line or word of handwritten text to extract characters.")

# 3. Model wrapper
class Batch:
    def __init__(self, imgs, line_mode=False):
        self.imgs = imgs
        self.line_mode = line_mode

@st.cache_resource(show_spinner="Loading SimpleHTR neural network weights...")
def load_htr_model():
    from main import DecoderType, Model, char_list_from_file
    char_list = char_list_from_file()
    # BestPath avoids beam search initialization overhead on cloud
    decoder_type = DecoderType.BestPath
    model = Model(char_list, decoder_type, must_restore=True)
    return model

# Only trigger model loading when needed, or with an explicit status container
status_box = st.empty()
status_box.info("Checking model readiness...")

try:
    model = load_htr_model()
    status_box.success("Model loaded and ready!")
except Exception as e:
    status_box.error(f"Error loading model weights: {e}")
    st.stop()

# 4. File uploader
uploaded_file = st.file_uploader(
    "Choose a handwritten image...", 
    type=["png", "jpg", "jpeg"]
)

st.sidebar.header("Inference Settings")
apply_denoise = st.sidebar.checkbox("Auto-clean background & ruled lines", value=False)

def preprocess_image(pil_img, clean=False):
    from preprocessor import Preprocessor
    img_gray = np.array(pil_img.convert("L"))

    if clean:
        blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        clean_img = cv2.subtract(thresh, detected_lines)
        img_gray = cv2.bitwise_not(clean_img)

    preprocessor = Preprocessor(img_size=(256, 32), dynamic_width=True, padding=16)
    return preprocessor.process_img(img_gray)

if uploaded_file is not None:
    raw_image = Image.open(uploaded_file)
    st.image(raw_image, caption="Uploaded Image")
    
    if st.button("Recognize Text", type="primary"):
        with st.spinner("Decoding handwriting..."):
            processed_img = preprocess_image(raw_image, clean=apply_denoise)
            batch = Batch([processed_img], line_mode=True)
            recognized, probability = model.infer_batch(batch)
            
            recognized_text = recognized[0] if recognized else ""
            conf_display = f"{probability[0]:.4f}" if (probability and probability[0] is not None) else "N/A"

        st.subheader("Extracted Text:")
        st.code(recognized_text, language="text")
        st.metric(label="Model Confidence Score", value=conf_display)