import os
import pandas as pd
from extract_players_positions import ExtractPlayersPositions

def generate_all_videos_players_positions(path_to_videos, details_csv_path):
    for dirpath, dirnames, filenames in os.walk(path_to_videos):
        for filename in [f for f in filenames if f.endswith(".mp4")]:
            generate_video_players_positions(os.path.join(dirpath, filename), details_csv_path)

def generate_video_players_positions(video_file_path, details_csv_path):
    base=os.path.basename(video_file_path)
    file = os.path.splitext(base)[0]
    dir = os.path.dirname(os.path.abspath(video_file_path))

    match_path_idx = video_file_path.find('match')
    match_path = video_file_path[match_path_idx:]

    print(match_path)
    video_details = pd.read_csv(details_csv_path)
    details = video_details[video_details['video_path'] == match_path]

    epp = ExtractPlayersPositions(video_file_path, match_path)
    epp.SetCourtCorners(
        [details.iloc[0]['ul_corner_y'], details.iloc[0]['ul_corner_x']],
        [details.iloc[0]['ur_corner_y'], details.iloc[0]['ur_corner_x']],
        [details.iloc[0]['br_corner_y'], details.iloc[0]['br_corner_x']],
        [details.iloc[0]['bl_corner_y'], details.iloc[0]['bl_corner_x']]
    )

    epp.Run(every_n_frames=1)

def merge_video_players_positions(path_to_videos, out_path):
    frames = []
    for dirpath, dirnames, filenames in os.walk(path_to_videos):
        for filename in [f for f in filenames if f.endswith("_players.csv")]:
            frames.append(pd.read_csv(os.path.join(dirpath, filename)))

    result = pd.concat(frames)
    pd.to_csv(out_path)
    return result

if __name__ == '__main__':
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    videos_path = f'{cur_dir}/../../raw_data/01_TRAIN'

    video_details_path = f'{cur_dir}/../../raw_data/video_details.csv'
    generate_all_videos_players_positions(videos_path, video_details_path)

    #merge_video_players_positions(path_to_videos, out_path)
