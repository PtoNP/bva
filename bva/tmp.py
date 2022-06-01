def get_shots_sequences(df):
    videos = df_shots['video_path'].unique()

    # init sequences for RNN model
    sequences = []
    targets = []

    # for all distinct videos
    for video in videos:
        # get video frames
        all_video_frames = df_shots[df_shots['video_path'] == video]

        current_shot_frames_features = []

        # loop on each frame 
        for idx, frame in all_video_frames.iterrows():
            features = [frame.birdie_x,
                        frame.birdie_y,
                        frame.birdie_dist_ul_corner,
                        frame.birdie_dist_ur_corner,
                        frame.birdie_dist_br_corner,
                        frame.birdie_dist_bl_corner,
                        frame.birdie_dist_left_net,
                        frame.birdie_dist_right_net]
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
                    targets.append(target)
                    current_shot_frames_features.append(features)
            # birdie not hit
            else:
                # a stroke has started ?
                if current_shot_frames_features:
                    current_shot_frames_features.append(features)
                # stroke not started ?
                else:
                    pass

        sequences.append(current_shot_frames_features)
    return np.array(sequences), np.array(targets)
