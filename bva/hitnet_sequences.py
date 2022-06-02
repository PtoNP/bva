import pandas as pd
import numpy as np
import os
from preprocess import get_features
import params

def get_video_sequences(video_frames):
    video_frames = video_frames.reset_index()

    sequences = []
    targets = []

    sequence_features = []
    idx = 0
    while idx < len(video_frames):
        if video_frames["birdie_hit"][idx:idx+12].sum() == 0:
            for ind, frame in video_frames[idx:idx+12].iterrows():
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
                sequence_features.append(features)
            sequences.append(sequence_features)
            targets.append(0)
            # continue ?
            if len(video_frames[idx+12:])>= 12:
                sequence_features = []
                idx += 12
            else:
                break
        else:
            id_hit = video_frames[idx:idx+12].index[video_frames["birdie_hit"][idx:idx+12]==1].to_list()[0]
            # start of sequence is below index 0 => try loop at next index
            if id_hit-6<0:
                idx +=1
                continue
            for ind, frame in video_frames[id_hit-6:id_hit+6].iterrows():
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
                sequence_features.append(features)
            sequences.append(sequence_features)
            targets.append(video_frames["birdie_hit"][id_hit])
            # continue ?
            if len(video_frames[id_hit+6:])>= 12:
                sequence_features = []
                idx = id_hit+6
            else:
                break

    return np.array(sequences), np.array(targets)

def get_sequences_by_video(df_url, vid_url, play_url, nb_videos_test):
    df = pd.read_csv(df_url)
    video_details = pd.read_csv(vid_url)
    play_details = pd.read_csv(play_url)
    df = df.merge(video_details, on='video_path')
    df = df.merge(play_details, on=['video_path', 'frame'])
    df_shots = get_features(df)

    videos = df_shots["video_path"].unique()
    all_videos_sequences = []
    all_videos_targets = []
    test_dict = {}

    counter = 0
    for video in videos:

        all_video_frames = df_shots[df_shots["video_path"]==video]
        X, y = get_video_sequences(all_video_frames)

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


if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    all_videos_sequences, all_videos_targets, test_dict = get_sequences_by_video(
                                        f'{cur_dir}/data/clean_dataset.csv',
                                        f'{cur_dir}/data/video_details.csv',
                                        f'{cur_dir}/data/players_positions.csv',
                                        2)
    unique, counts = np.unique(all_videos_targets, return_counts=True)
    print(all_videos_sequences.shape, all_videos_targets.shape, f"balance: {unique, counts}")
    print(len(test_dict))
