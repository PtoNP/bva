import os
import params
from strokenet_sequences import get_all_videos_sequences
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
from tensorflow.keras.models import Sequential, save_model, load_model
from strokenet_sequences import get_X_from_hitnet_output

# Get all sequences by hits + test_dict
def classif_get_data(with_net_features=False):
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    all_videos_sequences, all_videos_targets, \
        test_dict = get_all_videos_sequences(f'{cur_dir}/data/clean_dataset.csv',
                                            f'{cur_dir}/data/video_details.csv',
                                            f'{cur_dir}/data/players_positions.csv',
                                            2,
                                            with_net_features)
    return all_videos_sequences, all_videos_targets, test_dict

# Encode target + to_cat
def process_features_target(y):
    # cat y
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)
    return to_categorical(y_enc)

# 2class model creation
def classif_model(with_net_features=False):
    model = Sequential()
    if with_net_features:
        model.add(layers.Masking(mask_value=-1000, input_shape=(50,25)))
    else:
        model.add(layers.Masking(mask_value=-1000, input_shape=(50,21)))
    model.add(layers.GRU(units=64, activation='tanh', return_sequences=True))
    model.add(layers.GRU(units=32, activation='tanh', return_sequences=True))
    model.add(layers.GRU(units=24, activation='tanh', return_sequences=False))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dropout(rate=0.2))
    model.add(layers.Dense(30, activation='relu'))
    model.add(layers.Dropout(rate=0.2))
    model.add(layers.Dense(2, activation='softmax'))
    # Compilation
    model.compile(loss='binary_crossentropy',
                optimizer=RMSprop(learning_rate=0.001),
                metrics=["accuracy"])
    return model

# 2class model fitting
def classif_fitting(model, X_train, y_train, X_val, y_val):
    es = EarlyStopping(patience=30, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), \
        epochs=500, batch_size=16, callbacks=[es])
    return model, history

# 2class model training
def classif_training(filename):

    with_net_features = '_nets' in filename

    X, y, test_dict = classif_get_data(with_net_features)
    y_train_cat = process_features_target(y)
    X_train, X_val, y_train, y_val = train_test_split(X, y_train_cat, test_size=0.2)
    model = classif_model(with_net_features)
    model, history = classif_fitting(model, X_train, y_train, X_val, y_val)
    save_model(model, filename)
    return model, history, test_dict

# 2class predict
def predict_classes(hitnet_pred, predict_path, video_details_path, players_positions_path, mod_url):

    with_net_features = '_nets' in mod_url

    X_test = get_X_from_hitnet_output(hitnet_pred, predict_path,
                                      players_positions_path, video_details_path,
                                      with_net_features)
    model = load_model(mod_url)
    y_pred = model.predict(X_test)

    return y_pred


if __name__ == "__main__":
    classif_training()
    print("2class training done")
