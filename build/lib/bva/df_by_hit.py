import pandas as pd
import numpy as np
import os
from preprocess import get_features

def get_shots_sequences(df_url, vid_url, shots_test=300):
    df = pd.read_csv(df_url)
    video_details = pd.read_csv(vid_url)
    df = df.merge(video_details, on='video_path')
    df_shots = get_features(df)

    sequences = []
    targets = []
    current_shot_frames_features = []

    # loop on each frame 
    for idx, frame in df_shots.iterrows():
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
        # birdie is just hit
        if frame['birdie_hit'] == 1:
            target = frame['stroke']
            # new hit (end of previous stroke) ?
            if current_shot_frames_features:
                # backup all these frame as sequence
                sequences.append(current_shot_frames_features)
                #start new sequence
                current_shot_frames_features = []
                current_shot_frames_features.append(features)
                targets.append(target)
            # or first hit of the sequence ?
            else:
                current_shot_frames_features.append(features)
                targets.append(target)
        # birdie not hit
        else:
            # a stroke has started ?
            if current_shot_frames_features:
                current_shot_frames_features.append(features)
            # stroke not started ?
            else:
                pass
    sequences.append(current_shot_frames_features)

    seq_train = sequences[:-shots_test]
    targ_train = targets[:-shots_test]
    seq_tests = sequences[-shots_test:]
    targ_tests = targets[-shots_test:]

    return seq_train, targ_train, seq_tests, targ_tests


if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    seq_train, targ_train, seq_tests, targ_tests = get_shots_sequences(
                                        f'{cur_dir}/data/clean_dataset.csv',
                                        f'{cur_dir}/data/video_details.csv')

    print(len(seq_train), len(targ_train), len(seq_tests), len(targ_tests))
