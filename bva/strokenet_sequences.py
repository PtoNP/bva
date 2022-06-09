import pandas as pd
import numpy as np
import os
from preprocess import get_features, add_stroke_cat_to_dataset
from analyze_predicts import find_final_predict_from_hitnet
import params

def get_video_sequences_by_hit(video_frames, for_predict=False, with_net_features=False):
    video_frames = video_frames.reset_index()

    if for_predict:
        video_frames['stroke_cat'] = 'none'

    video_sequences = []
    video_targets = []
    window_frames_features = []

    sequence_started = False

    # loop on each frame 
    for idx, frame in video_frames.iterrows():
        target = frame.stroke_cat

        if with_net_features:
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
                        frame.left_net_x_nrm,
                        frame.left_net_y_nrm,
                        frame.right_net_x_nrm,
                        frame.right_net_y_nrm,
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
        else:
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

        if frame['birdie_hit'] == 1:
            if sequence_started:
                shape = np.shape(window_frames_features)
                padded_array = [[-1000] * np.shape(window_frames_features[0])[0]] * 200
                padded_array[:shape[0]] = window_frames_features

                video_sequences.append(padded_array[:50])
                video_targets.append(sequence_target)
                window_frames_features = []
            else:
                sequence_started = True

            sequence_target = target
            window_frames_features.append(features)
        else:
            if sequence_started:
                window_frames_features.append(features)

    # append last current sequence
    shape = np.shape(window_frames_features)
    padded_array = [[-1000] * np.shape(window_frames_features[0])[0]] * 200
    padded_array[:shape[0]] = window_frames_features

    video_sequences.append(padded_array[:50])
    video_targets.append(sequence_target)

    return np.array(video_sequences), np.array(video_targets)

def get_all_videos_sequences(clean_dataset_path, video_details_path,
                        players_path, nb_videos_test, with_net_features):
    clean_df = pd.read_csv(clean_dataset_path)
    clean_df = add_stroke_cat_to_dataset(clean_df)

    video_details = pd.read_csv(video_details_path)
    players_positions = pd.read_csv(players_path)
    clean_df = clean_df.merge(video_details, on='video_path')
    clean_df = clean_df.merge(players_positions, on=['video_path', 'frame'])
    df_shots = get_features(clean_df)

    videos = df_shots["video_path"].unique()
    all_videos_sequences = []
    all_videos_targets = []
    test_dict = {}

    counter = 0
    for video in videos:
        all_video_frames = df_shots[df_shots["video_path"]==video]
        X, y = get_video_sequences_by_hit(all_video_frames, with_net_features)

        if counter > len(videos) - nb_videos_test - 1:
            test_dict[video] = (X,y)
        else:
            # add to results
            if len(all_videos_sequences) > 0:
                all_videos_sequences = np.vstack((all_videos_sequences, X))
                all_videos_targets = np.concatenate([all_videos_targets, y])
            else:
                all_videos_sequences = X
                all_videos_targets = y

        counter += 1

    return all_videos_sequences, all_videos_targets, test_dict


def get_X_from_hitnet_output(hitnet_csv_path,
                             tracknet_predict_path,
                             players_csv_path,
                             video_details_path,
                             with_net_features):
    filename = tracknet_predict_path.split('/')[-1].replace('.csv','.mp4')
    video_path = f'./input/{filename}'

    hitnet_predict = pd.read_csv(hitnet_csv_path)
    hitnet_predict = find_final_predict_from_hitnet(hitnet_csv_path,
                            params.FINAL_PREDICT_PROBA_THRESHOLD)
    hitnet_predict['video_path'] = video_path

    video_details = pd.read_csv(video_details_path)
    video_details['video_path'] = video_path

    video_birdie_positions = pd.read_csv(tracknet_predict_path)
    video_birdie_positions['video_path'] = video_path
    video_birdie_positions['stroke'] = 'none'

    players_positions = pd.read_csv(players_csv_path)
    players_positions["video_path"] = video_path

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

    all_video_frames = video_birdie_positions.merge(video_details, on='video_path')

    all_video_frames = all_video_frames.merge(players_positions, on=['video_path','frame'])

    all_video_frames = all_video_frames.merge(hitnet_predict, on=['video_path','frame'])

    all_video_frames = all_video_frames.rename(columns = {'hit':'birdie_hit'})

    all_video_frames = get_features(all_video_frames)

    X, y = get_video_sequences_by_hit(all_video_frames,for_predict=True, with_net_features)

    return X


if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    all_videos_sequences, all_videos_targets, test_dict = get_all_videos_sequences(
                                        f'{cur_dir}/data/clean_dataset.csv',
                                        f'{cur_dir}/data/video_details.csv',
                                        f'{cur_dir}/data/players_positions.csv',
                                        2)
    unique, counts = np.unique(all_videos_targets, return_counts=True)
    print(all_videos_sequences.shape, all_videos_targets.shape, f"balance: {unique, counts}")
    print(len(test_dict))

    X_for_predict = get_X_from_hitnet_output(
                             f'{cur_dir}/data/hitnet_predict_match9_1_07_11.csv',
                             f'{cur_dir}/data/match9_1_07_11_predict.csv',
                             f'{cur_dir}/data/match9_1_07_11_players.csv',
                             f'{cur_dir}/data/match9_1_07_11_details.csv')
    print(X_for_predict.shape)
