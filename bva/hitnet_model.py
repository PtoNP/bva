import os
import pandas as pd
import numpy as np
import params
from hitnet_sequences import get_sequences_by_video, get_X_from_tracknet_output
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import RMSprop

#get all sequences to train the model, return X, y to be trained + dict for testing
def hitnet_get_data():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    X, y, test_dict = get_sequences_by_video(
                                        f'{cur_dir}/data/clean_dataset.csv',
                                        f'{cur_dir}/data/video_details.csv',
                                        f'{cur_dir}/data/players_positions.csv',
                                        True)
    return X, y, test_dict

#RNN model creation
def hitnet_model():
    model = Sequential()
    # model.add(tf.keras.Input(shape=(3793, 12, 21)))
    model.add(layers.Dense(48, activation='relu'))
    model.add(layers.GRU(units=64, activation='tanh', return_sequences=True))
    model.add(layers.GRU(units=16, activation='tanh', return_sequences=False))
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))
    # Compilation
    model.compile(loss='categorical_crossentropy',
                    optimizer="adam",
                    metrics=["accuracy"])
    return model

# RNN model training
def hitnet_fitting(model, X_train, y_train, X_val, y_val):
    es = EarlyStopping(patience=30, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), \
        epochs=500, batch_size=16, callbacks=[es])
    return model, history

# Hitnet RNN training
def hitnet_training(filename='/bva/models/hitnet'):
    X, y, test_dict = hitnet_get_data()
    y_cat = to_categorical(y)
    X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2)
    model = hitnet_model()
    model, history = hitnet_fitting(model, X_train, y_train, X_val, y_val)
    save_model(model, filename)
    return model, history, test_dict


def hitnet_predict_shots(predict_path, video_details_path, players_positions_path, mod_url):
    X_test = get_X_from_tracknet_output(predict_path, video_details_path,
                                        players_positions_path,
                                        params.NB_FRAMES)
    model = load_model(mod_url)
    y_pred = model.predict(X_test)

    return y_pred


if __name__ == "__main__":
    hitnet_training()
    print("Hitnet training done")
