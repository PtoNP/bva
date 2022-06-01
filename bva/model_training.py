import os
from preprocess import get_all_videos_sequences_by_window
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


def get_data(FRAMES_PER_WINDOW=5, NB_VIDEO_TEST=5):
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    df, X, y, test_dict = get_all_videos_sequences_by_window(
                    f'{cur_dir}/data/video_details.csv',
                    f'{cur_dir}/data/clean_dataset.csv', FRAMES_PER_WINDOW, NB_VIDEO_TEST)
    return df, X, y, test_dict

def process_target(y):
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)
    return to_categorical(y_enc)

def test_split(X_all, y_cat):
    X_train, X_val, y_train, y_val = train_test_split(X_all, y_cat, test_size=0.2)
    return X_train, X_val, y_train, y_val

def init_model():
    model = Sequential()
    # model.add(layers.Masking(mask_value=-1000, input_shape=(50,2))) => no padding
    model.add(layers.GRU(units=30, activation='tanh', return_sequences=True))
    model.add(layers.GRU(units=20, activation='tanh', return_sequences=False))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dropout(rate=0.2))
    model.add(layers.Dense(14, activation='softmax'))
    # Compilation
    model.compile(loss='categorical_crossentropy',
                optimizer=RMSprop(learning_rate=0.001),
                metrics=["accuracy"])
    return model

# Init & fitting
def init_fitting(model, X_train, y_train, X_val, y_val):
    es = EarlyStopping(patience=30, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), \
        epochs=500, batch_size=16, callbacks=[es])
    return model

if __name__ == "__main__":
    df, X, y, test_dict = get_data()
    y_cat = process_target(y)
    X_train, X_val, y_train, y_val = test_split(X, y_cat)
    model = init_model()
    history = init_fitting(model, X_train, y_train, X_val, y_val)
    print("ok done")
