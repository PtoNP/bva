import pandas as pd
import cv2
import os
import numpy as np
from players_positions.generate_output import generate_hitmap
from analyze_predicts import find_final_predict_from_hitnet
import params

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

def prepare_canvas(frame, hitmap):
    # resize scene frame
    scene = frame.copy()
    scene = image_resize(scene, height=600)
    hitmap = hitmap.copy()
    hitmap = image_resize(hitmap, height=600)
    # create black image
    canvas = np.zeros((scene.shape[0], scene.shape[1] + hitmap.shape[1],3), np.uint8)
    # pase image into canvas
    #print(scene.shape)
    canvas[0:scene.shape[0],0:scene.shape[1]] = scene
    canvas[0:scene.shape[0],scene.shape[1]:] = hitmap
    return canvas, scene, hitmap

def apply_ratio(original_image, resized_image, birdie_xy):
    rx = resized_image.shape[1] / original_image.shape[1]
    ry = resized_image.shape[0] / original_image.shape[0]

    return (int(birdie_xy['X'] * rx), int(birdie_xy['Y'] * ry))


def generate(   input_video_path, 
                birdie_csv_path,
                players_positions_path, 
                hitnet_predict_path,
                output_path):
    cap = cv2.VideoCapture(input_video_path)
    birdie_positions = pd.read_csv(birdie_csv_path)

    hits_df = find_final_predict_from_hitnet(hitnet_predict_path,
                                    params.FINAL_PREDICT_PROBA_THRESHOLD)
    hitmaps = generate_hitmap(players_positions_path, hits_df)

    success, image = cap.read()
    canvas, scene, hitmap = prepare_canvas(image, hitmaps[0])
    size = (canvas.shape[1], canvas.shape[0])
    fps = cap.get(cv2.CAP_PROP_FPS)

    if input_video_path[-3:] == 'avi':
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    elif input_video_path[-3:] == 'mp4':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    else:
        print('usage: video type can only be .avi or .mp4')
        exit(1)

    out = cv2.VideoWriter(output_path, fourcc, fps, size)

    count = 0
    birdie_history = []
    while success and count < len(hitmaps):
        if count < len(birdie_positions):
            birdie = birdie_positions.loc[count]
            birdie = apply_ratio(image, scene, birdie)
            if len(birdie_history) > 4:
                birdie_history = birdie_history[-3:0]
            birdie_history.append(birdie)

        canvas, scene, hitmap = prepare_canvas(image, hitmaps[count])

        for h in birdie_history:
            cv2.circle(canvas, h, 5, (0,0,255), -1)

        out.write(canvas)

        count = count + 1
        success, image = cap.read()

    out.release()

if __name__ == '__main__':
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    test_video_path = f'{cur_dir}/../raw_data/01_TRAIN/match9/rally_video/1_07_11.mp4'
    birdie_csv_path = f'{cur_dir}/data/match9_1_07_11_predict.csv'
    players_csv_path = f'{cur_dir}/data/match9_1_07_11_players.csv'
    hitnet_csv_path = f'{cur_dir}/data/hitnet_predict_match9_1_07_11.csv'

    generate(test_video_path, birdie_csv_path, players_csv_path, hitnet_csv_path)
