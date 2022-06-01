import streamlit as st
import os. path


#User's input video
st.markdown ("# BADMINTON VIDEO AUGMENTATION")
video_input = st.file_uploader("Choose a video file",type=['mp4', 'mpeg'] )
save_path = '../input_data'

#Download
if video_input is not None :

    video_input.name = "video_input"
    with open(os.path.join(save_path,video_input.name),"wb") as f:
      f.write(video_input.getbuffer())
