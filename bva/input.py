import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2 as cv
import os. path
from PIL import Image
import pandas as pd
import uuid
import os

video_to_load = True

# Input user's video
st.markdown ("# BADMINTON VIDEO AUGMENTATION")
video_input = st.file_uploader("Choose a video file",type=['mp4', 'mpeg'] )

if 'video_input' in st.session_state:
    if video_input != st.session_state['video_input']:
        video_to_load = True
    else:
        video_to_load = False
else:
    video_to_load = True

save_path = './input_data'
tmp_path = None

if 'tmp_path' in st.session_state:
    tmp_path = st.session_state['tmp_path']
    img_path = os.path.join(tmp_path,"image_mask.jpg")

# Download video
if video_input is not None and video_to_load:

    tmp_path = f'{save_path}/{ uuid.uuid4()}'

    st.session_state['video_input'] = video_input
    st.session_state['tmp_path'] = tmp_path

    if not os.path.exists(tmp_path):
        os.mkdir(tmp_path)

    with open(os.path.join(tmp_path,"video_input.mp4"),"wb") as f:
        f.write(video_input.getbuffer())

    # Download image (400x600)
    if tmp_path is not None:
        img_path = os.path.join(tmp_path,"image_mask.jpg")
        if video_input is not None :
            capture = cv.VideoCapture(os.path.join(tmp_path,"video_input.mp4"))
            fps = capture.get(cv.CAP_PROP_FPS)
            st.session_state['fps'] = fps
            has_next, frame = capture.read()
            resized = cv.resize (frame, (600,400), interpolation = cv.INTER_AREA)
            cv.imwrite(img_path, resized)
            (h, w) = frame.shape[:2]
            coef_w = w/600
            coef_h = h/400
            st.session_state['w'] = w
            st.session_state['h'] = h
            st.session_state['coef_w'] = coef_w
            st.session_state['coef_h'] = coef_h



if tmp_path is not None:
    # Create a canvas
    # Specify canvas parameters in application
    # drawing_mode = st.sidebar.selectbox(
    #     "Drawing tool:", ("point", "freedraw", "line", "rect", "circle", "transform")
    # )
    # if drawing_mode == 'point':
    #     point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)

    # realtime_update = st.sidebar.checkbox("Update in realtime", True)

    # Create a canvas component
    canvas_result = st_canvas(
        background_image=Image.open(img_path) if img_path else None,
        update_streamlit=True,
        height=400,
        drawing_mode='point',
        point_display_radius=3,
        key="canvas")

    a = ['input missing']*6
    objects_show_df = pd.DataFrame({'X' : a, 'Y' : a},index = ['1.up left corner',
                                                            '2.up right corner',
                                                            '3.bottom right corner',
                                                            '4.bottom left corner',
                                                            '5.left net',
                                                            '6.right net'])


    # Upload the canvas' data
    df_path = os.path.join(tmp_path,"video_details_input.csv")
    if canvas_result.json_data is not None:

        fps = st.session_state['fps']
        w = st.session_state['w']
        h = st.session_state['h']
        coef_w = st.session_state['coef_w']
        coef_h = st.session_state['coef_h']

        objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow
        for col in objects.select_dtypes(include=['object']).columns:
            objects[col] = objects[col].astype("str")
        for i in range(6) :
            try :
                objects_show_df.iloc[i,0] = str(int(objects.iloc[i,4]*coef_w))
            except :
                objects_show_df.iloc[i,0] = "input missing"
            try :
                objects_show_df.iloc[i,1] = str(int(objects.iloc[i,5]*coef_h))
            except :
                objects_show_df.iloc[i,1] = "input missing"

        columns = ['video_path','fps','width','height','ul_corner_x','ul_corner_y','ur_corner_x',
            'ur_corner_y','br_corner_x','br_corner_y','bl_corner_x','bl_corner_y',
            'left_net_x','left_net_y','right_net_x','right_net_y']

        # st.dataframe(objects_show_df)

        list_head = ["video-input.mp4", fps, int(w), int(h) ]
        list_canvas = objects_show_df.T.unstack().to_list()
        list_final = list_head + list_canvas

        video_detail_input = pd.DataFrame ([list_final], index =[1], columns = columns)

        video_detail_input.to_csv(df_path)
