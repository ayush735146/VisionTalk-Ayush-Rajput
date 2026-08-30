"""Streamlit interface for VisionTalk.
BY AYUSH RAJPUT
"""
import streamlit as st
from pathlib import Path
from inference import caption_image

st.set_page_config(page_title="VisionTalk — BY AYUSH RAJPUT", page_icon="🖼️", layout="centered")
st.markdown("<h1 style='text-align:center'>BY AYUSH RAJPUT</h1>", unsafe_allow_html=True)
st.title("VisionTalk — Image Captioning with CNN + LSTM")
st.write("Upload an image and generate a natural-language caption using the trained VGG16 + LSTM model.")

uploaded = st.file_uploader("Choose an image", type=["jpg","jpeg","png"])
if uploaded:
    temp = Path("outputs/uploaded_image.jpg")
    temp.parent.mkdir(exist_ok=True)
    temp.write_bytes(uploaded.getbuffer())
    st.image(uploaded, caption="Input Image", use_container_width=True)
    if not Path("artifacts/caption_model.keras").exists():
        st.warning("Model weights are not present yet. Run the training workflow in README.md first.")
    else:
        with st.spinner("Generating caption..."):
            caption = caption_image(temp)
        st.success(caption)
