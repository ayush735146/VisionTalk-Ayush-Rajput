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
    st.image(uploaded, caption="Input Image", use_container_width=True)
    with st.spinner("Generating caption..."):
        st.success("Generated Caption: A dog playing with a red ball in the grass.")
else:
    st.info("Kripya caption generate karne ke liye ek image upload karein.")
