from process import process_video, mediaFileType, video_stats_timeline
from process import image, emo_freqtable, img_stats, video_stats_meanfreq, video_stats_pctg
import pandas as pd
import cv2 as cv
import streamlit as st

from PIL import Image
import tempfile
from pyautogui import hotkey
from time import time


# Session statesss: Trick is to split processing logic from the display logic/code
if 'done_flag' not in st.session_state:
    st.session_state.done_flag = False
if 'input_type' not in st.session_state:
    st.session_state.input_type = 0
if 'pil_img' not in st.session_state:
    st.session_state.pil_img = ''
if 'emo_tl' not in st.session_state:
    st.session_state.emo_tl = ''
if 'freq_tb' not in st.session_state:
    st.session_state.freq_tb = ''
if 'emo_percent' not in st.session_state:
    st.session_state.emo_percent = ''
if 'prev_upload' not in st.session_state:
    st.session_state.prev_upload = ''
if 'model_option' not in st.session_state:
    st.session_state.model_option = ''

st.title("Emotion Analysis")
st.write("Input :  images or videos of student faces in an online class.")
st.write("Output :  statistics of the students' emotions during the class.")
st.text('')

st.session_state.model_option = st.selectbox(
    'Select a model',
    ('FER', 'RetinaFace', 'Mediapipe'))

uploaded_file = st.file_uploader("Choose an image / video")
st.text('')

if st.button('Run'):
    

    input_path = uploaded_file.name
    
    if mediaFileType(input_path) == 1:    # Image input
        st.session_state.input_type = 1
        st.session_state.pil_img = Image.open(uploaded_file).convert('RGB') 
        faces, st.session_state.pil_img = image(st.session_state.pil_img, st.session_state.model_option)
        print(faces)
        eft = emo_freqtable(faces)
        st.session_state.freq_tb = img_stats(eft)

    if mediaFileType(input_path) == 2:   # Video input
        st.session_state.input_type = 2
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        start_time = time()
        ef_df, avg = process_video(tfile.name, input_path, st.session_state.model_option)
        end_time = time()
        st.session_state.emo_tl = video_stats_timeline(ef_df)
        st.session_state.emo_percent = video_stats_pctg(ef_df)
        st.session_state.freq_tb = video_stats_meanfreq(avg)
        
    
    else:   #   Wrong file type
        pass
    
    st.session_state.prev_upload = uploaded_file
    
    st.write("Time taken: ", round(end_time - start_time, 3), 'seconds')
    # st.metric("Time taken", str(round(end_time - start_time, 3)) + ' seconds')
    
    if st.session_state.input_type == 1:
        st.header("Image Results")
        st.image(st.session_state.pil_img)
        st.write("Emotions Distribution")
        st.dataframe(st.session_state.freq_tb) 
        
        st.text('')
        #Reset
        if st.button('Reset'):
            hotkey('f5') 


    elif st.session_state.input_type == 2:
        st.header("Video Results")
        st.plotly_chart(st.session_state.emo_tl)
        st.plotly_chart(st.session_state.emo_percent)
        st.plotly_chart(st.session_state.freq_tb) 
        st.text('')
        st.download_button(
            label="Download output images (zip file)",
            data="emo_video_output\\images.zip",
            file_name="images.zip",
            mime="application/zip",
            key='img'
            )
        st.download_button(
            label="Download output video (mp4 file)",
            data="emo_video_output\\video.mp4",
            file_name="video.mp4",
            mime="video/mp4",
            key='video'
            )
        
        st.text('')
        # Reset
        if st.button('Reset'):
            hotkey('f5') 




