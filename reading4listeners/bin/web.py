import streamlit as st
import cv2
import pdf2image
import numpy as np

#

st.title('Reading4Listeners')
st.subheader('A deep-learning powered accessibility application which turns pdfs into audio files. Featuring ocr improvement and tts with inflection!')
uploaded_file = st.file_uploader("Choose a file", ['pdf', 'txt', 'muse','epub'], accept_multiple_files=False)

if uploaded_file is not None:
    with open(f'{sesspath}/{uploaded_file.name}', 'wb') as f:
        f.write(uploaded_file.getvalue())

    st.subheader('Use the sliders on the right to position the start and end points')
    start_x = st.sidebar.slider("Top Corner X", value=24 if use_default_image else 50, min_value=0, max_value=opencv_image.shape[1], key='sx')
    start_y = st.sidebar.slider("Top Corner Y", value=332 if use_default_image else 100, min_value=0, max_value=opencv_image.shape[0], key='sy')
    finish_x = st.sidebar.slider("Bottom Corner X", value=309 if use_default_image else 100, min_value=0, max_value=opencv_image.shape[1], key='fx')
    finish_y = st.sidebar.slider("Bottom Corner Y", value=330 if use_default_image else 100, min_value=0, max_value=opencv_image.shape[0], key='fy')