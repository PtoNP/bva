import os
from preprocess import get_all_videos_sequences_by_window
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


def get_data():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    seq_train, targ_train, seq_tests, targ_tests = get_shots_sequences(
                    f'{cur_dir}/data/clean_dataset.csv',
                    f'{cur_dir}/data/video_details.csv')
    return seq_train, targ_train, seq_tests, targ_tests

def process_features_target(X, y):
    # cat y
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)
    # padding X
    X_pad = pad_sequences(X, dtype='float32', \
        padding='post', value=-1000, maxlen=35)
    return X_pad, to_categorical(y_enc)

def init_model():
    model = Sequential()
    model.add(layers.Masking(mask_value=-1000, input_shape=(35,12)))
    model.add(layers.GRU(units=64, activation='tanh', return_sequences=True))
    model.add(layers.GRU(units=32, activation='tanh', return_sequences=True))
    model.add(layers.GRU(units=24, activation='tanh', return_sequences=False))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dropout(rate=0.2))
    model.add(layers.Dense(30, activation='relu'))
    model.add(layers.Dropout(rate=0.2))
    model.add(layers.Dense(13, activation='softmax'))
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

# Training
def training(filename='bva_model_trained.sav'):
    seq_train, targ_train, seq_tests, targ_tests = get_data()
    X_train_pad, y_train_cat = process_features_target(seq_train, targ_train)
    X_test_pad, y_test_pad = process_features_target(seq_tests, targ_tests)
    X_train, X_val, y_train, y_val = train_test_split(X_train_pad, y_train_cat, test_size=0.2)
    model = init_model()
    model = init_fitting(model, X_train, y_train, X_val, y_val)
    pickle.dump(model, open(filename, 'wb'))
    return model, X_test_pad, y_test_pad

if __name__ == "__main__":
    training()
    print("ok done")
