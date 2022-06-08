import os
import pandas as pd
from players_positions.generate_players_positions import generate_video_players_positions
from hitnet_model import hitnet_predict_shots
from Two_class_model import predict_classes
from video_output import generate
import params

class BvaMain:
    def __init__(self, tmp_path, hitnet_model_name):
        self.tmp_path = tmp_path
        self.video_input_path =  f"{tmp_path}/match_video_input.mp4"
        self.video_details_path = f"{tmp_path}/video_details_input.csv"
        self.predict_csv_path = f"{tmp_path}/match_video_input_predict.csv"
        self.players_csv_path = f"{tmp_path}/match_video_input_players.csv"
        self.hitnet_model_path = f"/bva/models/{hitnet_model_name}"
        self.hitnet_probas_path = f"{tmp_path}/hitnet_probas.csv"
        self.output_path = f"{tmp_path}/match_video_output.mp4"
        self.strokenet_model_path = "/bva/models/2class"
        self.strokenet_probas_path = f"{tmp_path}/strokenet_probas.csv"

        self.tracknet_predict_path = os.path.abspath("../../TrackNetv2/3_in_3_out/predict3.py")
        self.tracknet_weight_path = os.path.abspath("../../TrackNetv2/3_in_3_out/model906_30")

    def run_tracknetv2(self):
        cmd = f"python3.7 {self.tracknet_predict_path} --video_name={self.video_input_path} --load_weights={self.tracknet_weight_path}"
        os.system(cmd)
        print('Tracknet done')

    def run_players_detection(self):
        generate_video_players_positions(self.video_input_path, self.video_details_path)
        print('Players positions done')

    def run_hitnet(self):
        y_pred = hitnet_predict_shots(
                        self.predict_csv_path,
                        self.video_details_path,
                        self.players_csv_path,
                        self.hitnet_model_path,
                         self.hitnet_model_path)
        hit_probas_df = pd.DataFrame(y_pred)
        hit_probas_df.index.name = "index"
        hit_probas_df.to_csv(self.hitnet_probas_path)
        print('HitNet done')

    def run_strokenet(self):
         #hitnet_pred, predict_path, video_details_path, players_positions_path, mod_url
        y_pred = predict_classes(
                    self.hitnet_probas_path,
                    self.predict_csv_path,
                    self.video_details_path,
                    self.players_csv_path,
                    self.strokenet_model_path)
        stroke_probas_df = pd.DataFrame(y_pred)
        stroke_probas_df.index.name = "index"
        stroke_probas_df.to_csv(self.strokenet_probas_path)
        print('StrokeNet done')

    def run_build_augmented_video(self):
        generate(self.video_input_path,
                    self.predict_csv_path,
                    self.players_csv_path,
                    self.hitnet_probas_path,
                    self.strokenet_probas_path,
                    self.output_path)
        print('Build Final Video done')
