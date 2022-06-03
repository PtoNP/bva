import os
# from preprocess import get_all_videos_sequences_by_window
import params
from hitnet_sequences import get_sequences_by_video
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
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


def get_data():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    X, y, test_dict = get_sequences_by_video(
                                        f'{cur_dir}/data/clean_dataset.csv',
                                        f'{cur_dir}/data/video_details.csv',
                                        f'{cur_dir}/data/players_positions.csv')
    return X, y, test_dict

def init_model():
    model = Sequential()
    # model.add(tf.keras.Input(shape=(3793, 12, 21)))
    model.add(layers.Dense(48, activation='relu'))
    model.add(layers.GRU(units=64, activation='tanh', return_sequences=True))
    model.add(layers.GRU(units=16, activation='tanh', return_sequences=False))
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.Dense(2, activation='sigmoid'))
    # Compilation
    model.compile(loss='categorical_crossentropy',
                    optimizer="adam",
                    metrics=["accuracy"])
    return model

# Init & fitting
def init_fitting(model, X_train, y_train, X_val, y_val):
    es = EarlyStopping(patience=30, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), \
        epochs=500, batch_size=16, callbacks=[es])
    return model, history

# Training
def training(filename='hitnet_trained.sav'):
    X, y, test_dict = get_data()
    y_cat = to_categorical(y)
    X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2)
    model = init_model()
    model, history = init_fitting(model, X_train, y_train, X_val, y_val)
    joblib.dump(model, filename)
    return model, history, test_dict

if __name__ == "__main__":
    training()
    print("ok done")
