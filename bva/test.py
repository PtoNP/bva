
# Display an image
if video_input is not None :
    capture = cv.VideoCapture(os.path.join(save_path,video_input.name))
    has_next, frame = capture.read()
    st.image(frame)


# Download video
if video_input is not None :
    with open(os.path.join(save_path,"video_input"),"wb") as f:
      f.write(video_input.getbuffer())


# Download image

#if video_input is not None :
#    capture = cv.VideoCapture(os.path.join(save_path,video_input.name))
#    has_next, frame = capture.read()
#   with open(os.path.join(save_path,"image_mask"),"wb") as f:
#        f.write(frame)






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
#bg_image = st.image(display_image())

realtime_update = st.sidebar.checkbox("Update in realtime", True)
