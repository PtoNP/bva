import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2 as cv
import os. path
from PIL import Image
import pandas as pd


# Input user's video
st.markdown ("# BADMINTON VIDEO AUGMENTATION")
video_input = st.file_uploader("Choose a video file",type=['mp4', 'mpeg'] )

# Download video
save_path = '../input_data'
if video_input is not None :
    with open(os.path.join(save_path,"video_input"),"wb") as f:
      f.write(video_input.getbuffer())

img_path = os.path.join(save_path,"image_mask.jpg")

# Download image
if video_input is not None :
    capture = cv.VideoCapture(os.path.join(save_path,"video_input"))
    has_next, frame = capture.read()
    cv.imwrite(img_path, frame)


# Create a canvas
# Specify canvas parameters in application
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("point", "freedraw", "line", "rect", "circle", "transform")
)
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
if drawing_mode == 'point':
    point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
#bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])

realtime_update = st.sidebar.checkbox("Update in realtime", True)



# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    background_image=Image.open(img_path) if img_path else None,
    update_streamlit=realtime_update,
    height=frame.shape[0],
    width=frame.shape[1],
    drawing_mode=drawing_mode,
    point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
    key="canvas",
)

# Do something interesting with the image data and paths
# if canvas_result.image_data is not None:
#     st.image(canvas_result.image_data)
# if canvas_result.json_data is not None:
#     objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow
#     for col in objects.select_dtypes(include=['object']).columns:
#         objects[col] = objects[col].astype("str")
#     st.dataframe(objects)
