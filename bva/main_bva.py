import os
import pandas as pd
import sys
from players_positions.generate_players_positions import generate_video_players_positions
from load_predict import predict_shots
from analyze_predicts import find_final_predict_from_hitnet
from generate_video_output import output_video
import params

path = sys.argv[1]
full_path = os.path.abspath(path)

# 1/ TrackNetV2 call in Shell
cmd = 'echo this is a test'
os.system(cmd)
predict_csv = f"{path}/match_predict.csv"


# 2/ Players Positions
video_input = f"{path}/match_video_input.mp4"
csv_details = f"{path}/video_details_input.csv"

## generate_video_players_positions(video_input, csv_details)


# 3/ HitNet
# Get hits probas
play_det_csv = f"{path}/match_video_input_players.csv"
y_pred = predict_shots(predict_csv, csv_details, play_det_csv, "hitnet_trained_2vidtests.sav")
hit_probas_df = pd.DataFrame(y_pred)
hit_probas_df.to_csv(f"{path}/hitnet_probas.csv")

# Get hits predictions
hits_preds_df = find_final_predict_from_hitnet(f"{path}/hitnet_predicts.csv",
                               params.FINAL_PREDICT_PROBA_THRESHOLD)
hits_preds_df.to_csv(f"{path}/hitnet_predicts.csv")
hitnets_csv = f"{path}/hitnet_predicts.csv"


# 4/ Build augmented video
output_video(video_input, predict_csv, play_det_csv, hitnets_csv)
