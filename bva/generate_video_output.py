import pandas as pd
import cv2
import os
import numpy as np

OUTPUT_WIDTH = 1000

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def prepare_canvas(frame):
    # resize scene frame
    scene = frame.copy()
    #print(scene.shape)
    scene = image_resize(scene, width=800)
    #print(scene.shape)
    # create black image
    canvas = np.zeros((scene.shape[0], OUTPUT_WIDTH,3), np.uint8)
    # pase image into canvas
    #print(scene.shape)
    canvas[0:scene.shape[0],0:scene.shape[1]] = scene
    return canvas, scene

def apply_ratio(original_image, resized_image, birdie_xy):
    rx = resized_image.shape[1] / original_image.shape[1]
    ry = resized_image.shape[0] / original_image.shape[0]

    return (int(birdie_xy['X'] * rx), int(birdie_xy['Y'] * ry))


def output_video(input_video_path, birdie_csv_path, strokes_csv_path):
    #print(input_video_path)
    cap = cv2.VideoCapture(input_video_path)
    birdie_positions = pd.read_csv(birdie_csv_path)
    strokes = pd.read_csv(strokes_csv_path)

    success, image = cap.read()
    canvas, scene = prepare_canvas(image)
    size = (canvas.shape[1], canvas.shape[0])
    fps = 30

    if input_video_path[-3:] == 'avi':
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    elif input_video_path[-3:] == 'mp4':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    else:
        print('usage: video type can only be .avi or .mp4')
        exit(1)

    out = cv2.VideoWriter(input_video_path[:-4]+'_output'+input_video_path[-4:], fourcc, fps, size)

    count = 0
    birdie_history = []
    stroke_history = []
    while success:
        if count < len(birdie_positions):
            birdie = birdie_positions.loc[count]
            birdie = apply_ratio(image, scene, birdie)
            if len(birdie_history) > 4:
                birdie_history = birdie_history[-3:0]
            birdie_history.append(birdie)

        if count < len(strokes):
            stroke = str(strokes.loc[count]['stroke'])
            if stroke != 'nan':
                if len(stroke_history) > 19:
                    stroke_history = stroke_history[1:]
                stroke_history.append(stroke)

        canvas, scene = prepare_canvas(image)

        for h in birdie_history:
            cv2.circle(canvas, h, 5, (0,0,255), -1)

        sh_counter = 0
        for s in stroke_history:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(canvas, s, (810, 50 + 20*sh_counter),font,0.6,(0,255,0), 1, cv2.LINE_AA)
            sh_counter  += 1
        out.write(canvas)

        count = count + 1
        success, image = cap.read()

    out.release()

if __name__ == '__main__':
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    test_video_path = f'{cur_dir}/../raw_data/01_TRAIN/match2/rally_video/1_00_02.mp4'
    birdie_csv_path = f'{cur_dir}/../raw_data/1_00_02_predict.csv'
    strokes_csv_path = f'{cur_dir}/../raw_data/1_00_02_strokes.csv'

    output_video(test_video_path, birdie_csv_path, strokes_csv_path)
