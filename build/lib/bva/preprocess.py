import pandas as pd
import numpy as np
import os

def normalize_x(birdie_visible, birdie_x, width):
    if birdie_visible == 1:
        return birdie_x / width
    else:
        return -1

def normalize_y(birdie_visible, birdie_y, height):
    if birdie_visible == 1:
        return birdie_y / height
    else:
        return -1

def distance_to_court_point(birdie_visible, birdie_x, birdie_y, court_point_x, court_point_y):
    if birdie_visible == 1:
        return ((court_point_x - birdie_x)**2 + (court_point_y - birdie_y)**2)**.5
    else:
        return -1

def get_features(df_shots):
    df_shots['birdie_x_nrm'] = df_shots.apply(
        lambda x: normalize_x(x.birdie_visible, x.birdie_x, x.width), axis=1)
    df_shots['birdie_y_nrm'] = df_shots.apply(
        lambda x: normalize_y(x.birdie_visible, x.birdie_y, x.height), axis=1)

    return df_shots

def get_video_sequences_by_window(all_video_frames, nb_frame_per_window):
    video_sequences = []
    video_targets = []
    window_frames_features = []

    # loop on each frame 
    for idx, frame in all_video_frames.iterrows():
        features = [frame.birdie_visible,
                    frame.birdie_x_nrm,
                    frame.birdie_y_nrm,
                    frame.ul_corner_x,
                    frame.ul_corner_y,
                    frame.ur_corner_x,
                    frame.ur_corner_y,
                    frame.br_corner_x,
                    frame.br_corner_y,
                    frame.bl_corner_x,
                    frame.bl_corner_y]
        target = frame.stroke
        if frame.frame > nb_frame_per_window-1:
            video_sequences.append(window_frames_features)
            video_targets.append(target)
            window_frame_features = window_frames_features[1:]
            window_frame_features.append(features)
        else:
            window_frames_features.append(features)

    return np.array(video_sequences), np.array(video_targets)

def shift_y(y, nb_frames_per_window):
    to_insert = ['nan'] * nb_frames_per_window
    y = np.insert(y, 0, to_insert)
    # replace 'nan' value by 'no_hit'
    y[y == 'nan'] = 'no_hit'
    return y[nb_frames_per_window:]

def get_all_videos_sequences_by_window(video_details_path, clean_dataset_path,
                                    nb_frames_per_window, nb_videos_test):
    all_videos_sequences = []
    all_videos_targets = []
    test_dict = {}

    video_details = pd.read_csv(video_details_path)
    df_shots = pd.read_csv(clean_dataset_path)
    df_shots = df_shots.merge(video_details, on='video_path')
    df_shots = get_features(df_shots)

    videos_train = df_shots['video_path'].unique()[0:-nb_videos_test]

    for video in videos_train:
        # get video frames
        all_video_frames = df_shots[df_shots['video_path'] == video]
        # get sequences of one video
        X, y = get_video_sequences_by_window(all_video_frames, nb_frames_per_window)
        # shift y
        y = shift_y(y, nb_frames_per_window)

        # add to results
        if len(all_videos_sequences) > 0:
            all_videos_sequences = np.vstack((all_videos_sequences, X))
            all_videos_targets = np.concatenate([all_videos_targets, y])
        else:
            all_videos_sequences = X
            all_videos_targets = y

    videos_test = df_shots['video_path'].unique()[nb_videos_test:]
    for video in videos_test:
        # get video frames
        all_video_frames = df_shots[df_shots['video_path'] == video]
        # get sequences of one video
        X, y = get_video_sequences_by_window(all_video_frames, nb_frames_per_window)
        # shift y
        y = shift_y(y, nb_frames_per_window)

        test_dict[video] = (X,y[nb_frames_per_window:])

    return df_shots, all_videos_sequences, all_videos_targets, test_dict

if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    df, X, y, test_dict = get_all_videos_sequences_by_window(
                        f'{cur_dir}/data/video_details.csv',
                        f'{cur_dir}/data/clean_dataset.csv', 5, 5)
