import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2 as cv
import os. path


#Input user's video
st.markdown ("# BADMINTON VIDEO AUGMENTATION")
video_input = st.file_uploader("Choose a video file",type=['mp4', 'mpeg'] )
save_path = '../input_data'



#Download video
if video_input is not None :
    video_input.name = "video_input"
    with open(os.path.join(save_path,video_input.name),"wb") as f:
      f.write(video_input.getbuffer())


#Display an image
if video_input is not None :
    capture = cv.VideoCapture(os.path.join(save_path,video_input.name))
    has_next, frame = capture.read()
    st.image(frame)

#Create a canvas
st.title("Drawable Canvas")
st.markdown("""
Draw on the canvas, get the image data back into Python !
* Doubleclick to remove the selected object when not in drawing mode
""")
st.sidebar.header("Configuration")
# Specify brush parameters and drawing mode
b_width = st.sidebar.slider("Brush width: ", 1, 100, 10)
#b_color = st.sidebar.beta_color_picker("Enter brush color hex: ")
#bg_color = st.sidebar.beta_color_picker("Enter background color hex: ", "#eee")
drawing_mode = st.sidebar.checkbox("Drawing mode ?", True)
# Create a canvas component
image_data = st_canvas(
    b_width, height=150, drawing_mode=drawing_mode, key="canvas")


# Do something interesting with the image data
#if image_data is not None:
#    st.image(image_data)
