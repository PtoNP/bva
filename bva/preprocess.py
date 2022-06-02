import pandas as pd
import numpy as np
import os
import params

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
    court_width = params.COURT_WIDTH
    court_height = params.COURT_HEIGHT

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
    df_shots['player_A_court_x_nrm'] = df_shots.apply(
        lambda x: normalize_x(x.player_A_visible, x.player_A_court_x, court_width), axis=1)
    df_shots['player_A_court_y_nrm'] = df_shots.apply(
        lambda x: normalize_x(x.player_A_visible, x.player_A_court_y, court_height), axis=1)
    df_shots['player_B_court_x_nrm'] = df_shots.apply(
        lambda x: normalize_x(x.player_B_visible, x.player_B_court_x, court_width), axis=1)
    df_shots['player_B_court_y_nrm'] = df_shots.apply(
        lambda x: normalize_x(x.player_B_visible, x.player_B_court_y, court_height), axis=1)
    df_shots['player_A_img_x_nrm'] = df_shots.apply(
        lambda x: normalize_x(x.player_A_visible, x.player_A_img_x, x.width), axis=1)
    df_shots['player_A_img_y_nrm'] = df_shots.apply(
        lambda x: normalize_y(x.player_A_visible, x.player_A_img_y, x.height), axis=1)
    df_shots['player_B_img_x_nrm'] = df_shots.apply(
        lambda x: normalize_x(x.player_B_visible, x.player_B_img_x, x.width), axis=1)
    df_shots['player_B_img_y_nrm'] = df_shots.apply(
        lambda x: normalize_y(x.player_B_visible, x.player_B_img_y, x.height), axis=1)

    return df_shots

def get_video_sequences_by_window(all_video_frames, nb_frame_per_window):

    stroke_classes = all_video_frames['stroke'].unique()
    stroke_classes = [item for item in stroke_classes if not(pd.isnull(item)) == True]

    print(stroke_classes)

    video_sequences = []
    video_targets = []
    window_frames_features = []

    sequence_started = False
    print(nb_frame_per_window)
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
        #print(target)
        if target in stroke_classes:
            #print(idx)
            sequence_started = True
            sequence_target = target
            sequence_features_counter = 0
            window_frames_features = []

        if sequence_started:
            if sequence_features_counter >= nb_frame_per_window:
                #print('test')
                sequence_started = False
                video_sequences.append(window_frames_features)
                video_targets.append(sequence_target)
            else:
                #print('test2')
                window_frames_features.append(features)

            sequence_features_counter += 1
            #print(sequence_features_counter)

    return np.array(video_sequences), np.array(video_targets)

def clean_y(y):
    #replace 'nan' value by 'no_hit'
    y[y == 'nan'] = 'no_hit'
    return y

def multiply_targets(y, nb_frames_per_window):
    i = 0
    while i < len(y)-nb_frames_per_window:
        if y[i] != 'no_hit':
            for j in range(nb_frames_per_window):
                y[i+j+1] = y[i]
            i += nb_frames_per_window + 1
        else:
            i += 1
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
        #y = clean_y(y)
        #y = multiply_targets(y,nb_frames_per_window)

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
        #y = clean_y(y)
        #y = multiply_targets(y,nb_frames_per_window)

        test_dict[video] = (X,y)

    return df_shots, all_videos_sequences, all_videos_targets, test_dict



def get_video_sequences_for_predict(all_video_frames, nb_frame_per_window):

    sequences = []
    window_features = []
    all_features = []
    counter = 0
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
                    frame.bl_corner_y_nrm,
                    frame.player_A_visible,
                    frame.player_B_visible,
                    frame.player_A_court_x_nrm,
                    frame.player_A_court_y_nrm,
                    frame.player_B_court_x_nrm,
                    frame.player_B_court_y_nrm,
                    frame.player_A_img_x_nrm,
                    frame.player_A_img_y_nrm,
                    frame.player_B_img_x_nrm,
                    frame.player_B_img_y_nrm]
        all_features.append(features)

    while counter < len(all_features) - nb_frame_per_window:
        window_features = all_features[counter:counter+nb_frame_per_window]
        sequences.append(window_features)
        counter += 1

    return np.array(sequences)

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

    X = get_video_sequences_for_predict(video_birdie_positions, nb_frames_per_window)

    return X


if __name__ == "__main__":
    FRAMES_PER_WINDOW = 5
    NB_VIDEO_TEST = 2

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    # df, X, y, test_dict = get_all_videos_sequences_by_window(
    #                     f'{cur_dir}/data/video_details.csv',
    #                     f'{cur_dir}/data/clean_dataset.csv', FRAMES_PER_WINDOW, NB_VIDEO_TEST)

    #print(y[0:30])
    #print(test_dict['match9/rally_video/1_07_10.mp4'][0].shape)
    #print(test_dict['match9/rally_video/1_07_10.mp4'][1].shape)
    #print(test_dict['match9/rally_video/1_07_10.mp4'][1])

    predict_path = f'{cur_dir}/../raw_data/1_00_02_predict.csv'
    video_details_path = f'{cur_dir}/../raw_data/1_00_02_details.csv'

    X_test = get_X_from_tracknet_output(predict_path, video_details_path, FRAMES_PER_WINDOW)
    #print(X_test.shape)
