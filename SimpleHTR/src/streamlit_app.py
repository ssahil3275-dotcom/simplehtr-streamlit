import os
import sys
from pathlib import Path

# Add script directory and its parent directories to sys.path
current_dir = Path(__file__).resolve().parent
for parent in [current_dir, current_dir.parent, current_dir.parent.parent]:
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))

# Ensure working directory is set to src
try:
    os.chdir(str(current_dir))
except Exception:
    pass

import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(
    page_title="Handwriting Recognition (SimpleHTR)",
    page_icon="✍️",
    layout="centered"
)

st.title("✍️ Handwriting OCR Web Interface")
st.markdown("Upload a single line or word of handwritten text to extract characters.")

# Diagnostic check visible on the web UI
status_box = st.empty()

try:
    import main
    from main import DecoderType, Model, char_list_from_file
    from preprocessor import Preprocessor
except ModuleNotFoundError as e:
    status_box.error(
        f"Python cannot find 'main.py'.\n"
        f"Current working dir: {os.getcwd()}\n"
        f"Files in current folder: {os.listdir(os.getcwd())}\n"
        f"sys.path: {sys.path[:3]}"
    )
    st.stop()

class Batch:
    def __init__(self, imgs, line_mode=False):
        self.imgs = imgs
        self.line_mode = line_mode

@st.cache_resource(show_spinner="Loading SimpleHTR neural network weights...")
def load_htr_model():
    char_list = char_list_from_file()
    decoder_type = DecoderType.BestPath
    return Model(char_list, decoder_type, must_restore=True)

try:
    model = load_htr_model()
    status_box.success("Model loaded and ready!")
except Exception as e:
    status_box.error(f"Error loading model weights: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose a handwritten image...", 
    type=["png", "jpg", "jpeg"]
)

st.sidebar.header("Inference Settings")
apply_denoise = st.sidebar.checkbox("Auto-clean background & ruled lines", value=False)

def preprocess_image(pil_img, clean=False):
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
