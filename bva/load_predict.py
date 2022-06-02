import os
from hitnet_sequences import get_X_from_tracknet_output
import params
from df_by_hit import get_shots_sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences
from analyze_predicts import find_best_targets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import RMSprop
import joblib

def load_model(filename='hitnet_trained.sav'):
    loaded_model = joblib.load(filename)
    return loaded_model

def predict_shots(predict_path, video_details_path, players_positions_path):
    X_test = get_X_from_tracknet_output(predict_path, video_details_path,
                                        players_positions_path,
                                        params.NB_FRAMES)
    model = load_model()
    y_pred = model.predict(X_test)

    return y_pred

if __name__ == '__main__':
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    y_pred = predict_shots(f'{cur_dir}/data/match9_1_07_11_predict.csv',
                            f'{cur_dir}/data/match9_1_07_11_details.csv',
                            f'{cur_dir}/data/match9_1_07_11_players.csv')
    print(y_pred)
