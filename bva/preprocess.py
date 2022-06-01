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
    df_shots['ul_corner_x_nrm'] = df_shots.apply(
        lambda x: normalize_y(1, x.ul_corner_x, x.width), axis=1)
    df_shots['ul_corner_y_nrm'] = df_shots.apply(
        lambda x: normalize_y(1, x.ul_corner_y, x.height), axis=1)
    df_shots['ur_corner_x_nrm'] = df_shots.apply(
        lambda x: normalize_y(1, x.ur_corner_x, x.width), axis=1)
    df_shots['ur_corner_y_nrm'] = df_shots.apply(
        lambda x: normalize_y(1, x.ur_corner_y, x.height), axis=1)
    df_shots['br_corner_x_nrm'] = df_shots.apply(
        lambda x: normalize_y(1, x.br_corner_x, x.width), axis=1)
    df_shots['br_corner_y_nrm'] = df_shots.apply(
        lambda x: normalize_y(1, x.br_corner_y, x.height), axis=1)
    df_shots['bl_corner_x_nrm'] = df_shots.apply(
        lambda x: normalize_y(1, x.bl_corner_x, x.width), axis=1)
    df_shots['bl_corner_y_nrm'] = df_shots.apply(
        lambda x: normalize_y(1, x.bl_corner_y, x.height), axis=1)

    return df_shots

def get_video_sequences_by_window(all_video_frames, nb_frame_per_window):
    video_sequences = []
    video_targets = ['nan'] * nb_frame_per_window
    window_frames_features = []

    last_frame_idx = len(all_video_frames) - 1

    # loop on each frame 
    for idx, frame in all_video_frames.iterrows():
        features = [frame.birdie_visible,
                    frame.birdie_x_nrm,
                    frame.birdie_y_nrm,
                    frame.ul_corner_x_nrm,
                    frame.ul_corner_y_nrm,
                    frame.ur_corner_x_nrm,
                    frame.ur_corner_y_nrm,
                    frame.br_corner_x_nrm,
                    frame.br_corner_y_nrm,
                    frame.bl_corner_x_nrm,
                    frame.bl_corner_y_nrm]
        target = frame.stroke
        if frame.frame > nb_frame_per_window-2:
            video_sequences.append(window_frames_features)
            video_targets.append(target)
            window_frame_features = window_frames_features[1:]
            window_frame_features.append(features)
        else:
            window_frames_features.append(features)
            video_targets.append(target)

        if last_frame_idx == frame.frame - nb_frame_per_window:
            break

    nb_sequence = len(video_sequences)
    video_targets = video_targets[0:nb_sequence]

    return np.array(video_sequences), np.array(video_targets)

def clean_y(y):
    #replace 'nan' value by 'no_hit'
    y[y == 'nan'] = 'no_hit'
    return y

def get_all_videos_sequences_by_window(video_details_path, clean_dataset_path,
                                    nb_frames_per_window, nb_videos_test):
    all_videos_sequences = []
    all_videos_targets = []
    test_dict = {}

    video_details = pd.read_csv(video_details_path)
    df_shots = pd.read_csv(clean_dataset_path)
    df_shots = df_shots.merge(video_details, on='video_path')
    df_shots = get_features(df_shots)

    videos_train = df_shots['video_path'].unique()[:-nb_videos_test]

    for video in videos_train:
        # get video frames
        all_video_frames = df_shots[df_shots['video_path'] == video]
        # get sequences of one video
        X, y = get_video_sequences_by_window(all_video_frames, nb_frames_per_window)
        # clean y
        y = clean_y(y)

        # add to results
        if len(all_videos_sequences) > 0:
            all_videos_sequences = np.vstack((all_videos_sequences, X))
            all_videos_targets = np.concatenate([all_videos_targets, y])
        else:
            all_videos_sequences = X
            all_videos_targets = y

    videos_test = df_shots['video_path'].unique()[-nb_videos_test:]
    for video in videos_test:
        # get video frames
        all_video_frames = df_shots[df_shots['video_path'] == video]
        # get sequences of one video
        X, y = get_video_sequences_by_window(all_video_frames, nb_frames_per_window)
        # clean y
        y = clean_y(y)

        test_dict[video] = (X,y)

    return df_shots, all_videos_sequences, all_videos_targets, test_dict


def get_X_from_tracknet_output(predict_path, video_details_path, nb_frames_per_window):
    filename = predict_path.split('/')[-1].replace('.csv','.mp4')
    video_path = f'./input/{filename}'

    video_details = pd.read_csv(video_details_path)
    video_details['video_path'] = video_path

    video_birdie_positions = pd.read_csv(predict_path)
    video_birdie_positions['video_path'] = video_path
    video_birdie_positions['stroke'] = 'none'

    video_birdie_positions = video_birdie_positions.rename(
                        columns = {'Frame':'frame',
                                    'Visibility':'birdie_visible',
                                    'X': 'birdie_x',
                                    'Y':'birdie_y',
                                    'Time': 'time',
                                    'stroke': 'stroke'}) \
                            .reindex(columns = ['video_path', 'frame',
                                                'birdie_visible', 'birdie_x',
                                                'birdie_y',
                                                'time',
                                                'stroke'])

    video_birdie_positions = video_birdie_positions.merge(video_details, on='video_path')

    video_birdie_positions = get_features(video_birdie_positions)

    X, y = get_video_sequences_by_window(video_birdie_positions, nb_frames_per_window)

    return X


if __name__ == "__main__":
    FRAMES_PER_WINDOW = 5
    NB_VIDEO_TEST = 2

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    df, X, y, test_dict = get_all_videos_sequences_by_window(
                        f'{cur_dir}/data/video_details.csv',
                        f'{cur_dir}/data/clean_dataset.csv', FRAMES_PER_WINDOW, NB_VIDEO_TEST)

    #print(test_dict['match9/rally_video/1_07_10.mp4'][0].shape)
    #print(test_dict['match9/rally_video/1_07_10.mp4'][1].shape)
    #print(test_dict['match9/rally_video/1_07_10.mp4'][1])

    predict_path = f'{cur_dir}/../raw_data/1_00_02_predict.csv'
    video_details_path = f'{cur_dir}/../raw_data/1_00_02_details.csv'

    X_test = get_X_from_tracknet_output(predict_path, video_details_path, FRAMES_PER_WINDOW)
