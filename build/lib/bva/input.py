import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2 as cv
import os. path
from PIL import Image
import pandas as pd



# Input user's video
st.markdown ("# BADMINTON VIDEO AUGMENTATION")
video_input = st.file_uploader("Choose a video file",type=['mp4', 'mpeg'] )

save_path = '../input_data'

# Download video
if video_input is not None :
    with open(os.path.join(save_path,"video_input"),"wb") as f:
      f.write(video_input.getbuffer())


# Download image (400x600)
img_path = os.path.join(save_path,"image_mask.jpg")
if video_input is not None :
    capture = cv.VideoCapture(os.path.join(save_path,"video_input"))
    has_next, frame = capture.read()
    resized = cv.resize (frame, (600,400), interpolation = cv.INTER_AREA)
    cv.imwrite(img_path, resized)
    (w, h) = frame.shape[:2]
    coef_w = w/600
    coef_h = h/400


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
    height=400,
    drawing_mode=drawing_mode,
    point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
    key="canvas",
)

# Upload the canvas' data
df_path = os.path.join(save_path,"params_court_df.csv")
if canvas_result.json_data is not None:
    objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow
    #objects.columns['left', 'top'] = ['x','y']
    for col in objects.select_dtypes(include=['object']).columns:
        objects[col] = objects[col].astype("str")
    #params_courts_fd =  objects[]
    st.dataframe(objects['left'])

    objects.to_csv(df_path)




                #ul_corner_x', 'ul_corner_y',
                #ur_corner_x', 'ur_corner_y',
                #br_corner_x', 'br_corner_y',
                #bl_corner_x', 'bl_corner_y',
                #left_net_x', 'left_net_y',
                #right_net_x', 'right_net_y'
