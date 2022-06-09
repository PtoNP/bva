import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2 as cv
import os
from PIL import Image
import pandas as pd
import uuid
import os
import subprocess
import glob
from os.path import exists
from main_bva import BvaMain
import time
from analyze_predicts import find_final_predict_from_hitnet
import params

st.set_page_config(page_title="BVA", page_icon="🏸", layout="centered")

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

    with open(os.path.join(tmp_path,"match_video_input.mp4"),"wb") as f:
        f.write(video_input.getbuffer())

    # Download image (400x600)
    if tmp_path is not None:
        img_path = os.path.join(tmp_path,"image_mask.jpg")
        if video_input is not None :
            capture = cv.VideoCapture(os.path.join(tmp_path,"match_video_input.mp4"))
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

        list_head = ["match_video_input.mp4", fps, int(w), int(h) ]
        list_canvas = objects_show_df.T.unstack().to_list()
        list_final = list_head + list_canvas

        video_detail_input = pd.DataFrame ([list_final], index =[0], columns = columns)

        video_detail_input.to_csv(df_path)

    with st.expander("Parameters"):

        hitnet_values = {'hitnet' : 'Hitnet 1 (no sequence cleaning)',
                        'hitnet_mirror': 'Hitnet 1 + data mirroring',
                        'hitnet2': 'Hitnet 2 (with sequence cleaning)',
                        'hitnet_mirror2': 'Hitnet 2 + data mirroring'}
        hitnet_model_name = st.selectbox('Hitnet Model :',
            list(hitnet_values.keys()), format_func= lambda x : hitnet_values[x])

        params.FINAL_PREDICT_PROBA_THRESHOLD = st.slider(
                    'Hitnet predict threshold',
                    0.5, 1.0, params.FINAL_PREDICT_PROBA_THRESHOLD, 0.01)

        params.FINAL_PREDICT_MIN_FRAMES_BEFORE_NEXT_HIT = st.slider(
                    'Hitnet min frames between hits',
                    5, 20, params.FINAL_PREDICT_MIN_FRAMES_BEFORE_NEXT_HIT, 1)

        params.MIN_FRAMES_FOR_HIT = st.slider(
                    'Hitnet min hit frames',
                    1, 5, params.MIN_FRAMES_FOR_HIT, 1)

        params.REMOVE_DIRTY_SEQUENCES_AFTER_PREDICTION = st.checkbox('Remove dirty sequences after predict')

    if st.button('Start video augmentation'):
        print(f'FINAL_PREDICT_PROBA_THRESHOLD : {params.FINAL_PREDICT_PROBA_THRESHOLD}')
        print(f'FINAL_PREDICT_MIN_FRAMES_BEFORE_NEXT_HIT : {params.FINAL_PREDICT_MIN_FRAMES_BEFORE_NEXT_HIT}')
        print(f'MIN_FRAMES_FOR_HIT : {params.MIN_FRAMES_FOR_HIT}')
        print(f'REMOVE_DIRTY_SEQUENCES_AFTER_PREDICTION : {params.REMOVE_DIRTY_SEQUENCES_AFTER_PREDICTION}')

        bva = BvaMain(tmp_path, hitnet_model_name)
        bva.run_tracknetv2()
        bva.run_players_detection()
        bva.run_hitnet()
        bva.run_strokenet()
        bva.run_build_augmented_video()

        st.session_state['video_path'] = bva.video_input_path
        st.session_state['video_details_path'] = bva.video_details_path
        st.session_state['predict_csv'] = bva.predict_csv_path
        st.session_state['players_csv'] = bva.players_csv_path
        st.session_state['hit_probas_csv'] =  bva.hitnet_probas_path
        st.session_state['output_path'] = bva.output_path
        st.session_state['stroke_probas_csv'] =  bva.strokenet_probas_path

    if 'hit_probas_csv' in st.session_state and exists(st.session_state['hit_probas_csv']):
        df = pd.read_csv(st.session_state['hit_probas_csv'])

        hits_df = find_final_predict_from_hitnet(st.session_state['hit_probas_csv'],
                                    params.FINAL_PREDICT_PROBA_THRESHOLD)

        st.dataframe(df.merge(hits_df, left_on='index',right_on='frame'))

    video_to_show = None
    if 'output_path' in st.session_state and exists(st.session_state['output_path']):
        video_to_show = st.session_state['output_path']
        if video_to_show:
            dl_values = {st.session_state['output_path'] : ['Result video', 'video_output.mp4'],
                         st.session_state['video_details_path'] : ['Video details','video_details.csv'],
                         st.session_state['predict_csv'] : ['Tracknet csv','tracknet_output.csv'],
                         st.session_state['players_csv'] : ['Players csv','players_output.csv'],
                         st.session_state['hit_probas_csv'] : ['Hitnet csv', 'hitnet_output.csv'],
                         st.session_state['stroke_probas_csv'] : ['Strokenet csv','strokenet_output.csv']}
            dl_file = st.selectbox('File to download :',
                list(dl_values.keys()), format_func= lambda x : dl_values[x][0])

            if exists(os.path.abspath(dl_file)):
                with open(os.path.abspath(dl_file), 'rb') as v:
                    st.download_button('Download', v, file_name=dl_values[dl_file][1])
