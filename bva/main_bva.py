import os
import pandas as pd
import sys
from players_positions.generate_players_positions import generate_video_players_positions
from load_predict import predict_shots
from analyze_predicts import find_final_predict_from_hitnet
from generate_video_output import output_video
import params

path = sys.argv[1]
video_input = f"{path}/match_video_input.mp4"
csv_details = f"{path}/video_details_input.csv"

# 1/ TrackNetV2 call in Shell
tracknet_path = os.path.abspath("../../TrackNetv2/3_in_3_out/predict3.py")
weight_path = os.path.abspath("../../TrackNetv2/3_in_3_out/model906_30")
cmd = f"python {tracknet_path} --video_name={video_input} --load_weights={weight_path}"
os.system(cmd)
predict_csv = f"{path}/match_video_input_predict.csv"


# 2/ Players Positions
## generate_video_players_positions(video_input, csv_details)


# 3/ HitNet
# Get hits probas
# play_det_csv = f"{path}/match_video_input_players.csv"
# y_pred = predict_shots(predict_csv, csv_details, play_det_csv, "hitnet_trained_2vidtests.sav")
# hit_probas_df = pd.DataFrame(y_pred)
# hit_probas_df.index.name = "index"
# hit_probas_df.to_csv(f"{path}/hitnet_probas.csv")
# hit_probas_csv = f"{path}/hitnet_probas.csv"


# # 4/ Build augmented video
# output_video(video_input, predict_csv, play_det_csv, hit_probas_csv)
