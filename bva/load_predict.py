import os
from preprocess import get_all_videos_sequences_by_window, get_X_from_tracknet_output
from df_by_hit import get_shots_sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences
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

def predict_shots(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred
