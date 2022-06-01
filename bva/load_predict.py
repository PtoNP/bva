import os
from preprocess import get_all_videos_sequences_by_window, get_X_from_tracknet_output
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
import pickle

def load_model(filename='bva_model_trained.sav'):
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model


def predict_shots(predict_path, video_details_path):
    X_test = get_X_from_tracknet_output(predict_path, video_details_path,
                               params.NB_FRAME_PADDING)
    model = load_model()
    y_pred = model.predict(X_test)
    y_classes = find_best_targets(params.CLASSES, y_pred, params.THRESHOLD)

    return y_pred, y_classes

if __name__ == '__main__':
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    y_pred, y_classes = predict_shots(f'{cur_dir}/data/match9_1_07_11_predict.csv',
                  f'{cur_dir}/data/match9_1_07_11_details.csv')
    print(y_pred[:10], y_classes[:10])
