import pandas as pd
import numpy as np
import os
import params

def normalize_x(birdie_visible, birdie_x, width):
    if birdie_visible == 1:
        return birdie_x / width
    else:
        return -1

def normalize_y(birdie_visible, birdie_y, height):
    if birdie_visible == 1:
        return birdie_y / height
    else:
        return -1

def distance_to_court_point(birdie_visible, birdie_x, birdie_y, court_point_x, court_point_y):
    if birdie_visible == 1:
        return ((court_point_x - birdie_x)**2 + (court_point_y - birdie_y)**2)**.5
    else:
        return -1

def get_features(df_shots):
    court_width = params.COURT_WIDTH
    court_height = params.COURT_HEIGHT

    df_shots['birdie_x_nrm'] = df_shots.apply(
        lambda x: normalize_x(x.birdie_visible, x.birdie_x, x.width), axis=1)
    df_shots['birdie_y_nrm'] = df_shots.apply(
        lambda x: normalize_y(x.birdie_visible, x.birdie_y, x.height), axis=1)
    df_shots['ul_corner_x_nrm'] = df_shots.apply(
        lambda x: normalize_y(1, x.ul_corner_x, x.width), axis=1)
    df_shots['ul_corner_y_nrm'] = df_shots.apply(
        lambda x: normalize_y(1, x.ul_corner_y, x.height), axis=1)
    df_shots['ur_corner_x_nrm'] = df_shots.apply(
        lambda x: normalize_y(1, x.ur_corner_x, x.width), axis=1)
    df_shots['ur_corner_y_nrm'] = df_shots.apply(
        lambda x: normalize_y(1, x.ur_corner_y, x.height), axis=1)
    df_shots['br_corner_x_nrm'] = df_shots.apply(
        lambda x: normalize_y(1, x.br_corner_x, x.width), axis=1)
    df_shots['br_corner_y_nrm'] = df_shots.apply(
        lambda x: normalize_y(1, x.br_corner_y, x.height), axis=1)
    df_shots['bl_corner_x_nrm'] = df_shots.apply(
        lambda x: normalize_y(1, x.bl_corner_x, x.width), axis=1)
    df_shots['bl_corner_y_nrm'] = df_shots.apply(
        lambda x: normalize_y(1, x.bl_corner_y, x.height), axis=1)
    df_shots['left_net_x_nrm'] = df_shots.apply(
        lambda x: normalize_x(1, x.left_net_x, x.width), axis=1)
    df_shots['left_net_y_nrm'] = df_shots.apply(
        lambda x: normalize_y(1, x.left_net_y, x.width), axis=1)
    df_shots['right_net_x_nrm'] = df_shots.apply(
        lambda x: normalize_x(1, x.right_net_x, x.width), axis=1)
    df_shots['right_net_y_nrm'] = df_shots.apply(
        lambda x: normalize_y(1, x.right_net_y, x.width), axis=1)
    df_shots['player_A_court_x_nrm'] = df_shots.apply(
        lambda x: normalize_x(x.player_A_visible, x.player_A_court_x, court_width), axis=1)
    df_shots['player_A_court_y_nrm'] = df_shots.apply(
        lambda x: normalize_x(x.player_A_visible, x.player_A_court_y, court_height), axis=1)
    df_shots['player_B_court_x_nrm'] = df_shots.apply(
        lambda x: normalize_x(x.player_B_visible, x.player_B_court_x, court_width), axis=1)
    df_shots['player_B_court_y_nrm'] = df_shots.apply(
        lambda x: normalize_x(x.player_B_visible, x.player_B_court_y, court_height), axis=1)
    df_shots['player_A_img_x_nrm'] = df_shots.apply(
        lambda x: normalize_x(x.player_A_visible, x.player_A_img_x, x.width), axis=1)
    df_shots['player_A_img_y_nrm'] = df_shots.apply(
        lambda x: normalize_y(x.player_A_visible, x.player_A_img_y, x.height), axis=1)
    df_shots['player_B_img_x_nrm'] = df_shots.apply(
        lambda x: normalize_x(x.player_B_visible, x.player_B_img_x, x.width), axis=1)
    df_shots['player_B_img_y_nrm'] = df_shots.apply(
        lambda x: normalize_y(x.player_B_visible, x.player_B_img_y, x.height), axis=1)

    return df_shots

def add_stroke_cat_to_dataset(clean_df):

    def set_att_def(x):
        if x in ['short_serve','drop','net_shot','drive','half_smash','full_smash','net_kill']:
            return 'att'
        if x in ['long_serve','lob','clear','short_def','lift','long_def']:
            return 'def'
        return 'no_hit'

    clean_df['stroke_cat'] = clean_df.apply(lambda x: set_att_def(x['stroke']), axis=1)
    return clean_df

def mirror_data(df_url, vid_url, play_url):
    court_width = params.COURT_WIDTH

    df = pd.read_csv(df_url)
    video_details = pd.read_csv(vid_url)
    df = df.merge(video_details, on='video_path')
    play_details = pd.read_csv(play_url)
    df = df.merge(play_details, on=['video_path', 'frame'])
    print(df.shape)
    mirror_df = df.copy()
    print(mirror_df.shape)

    mirror_df['video_path'] = mirror_df.apply(lambda x: x.video_path[:-4]+"_mirr.mp4", axis=1)
    mirror_df['birdie_x'] = mirror_df.apply(lambda x: x.birdie_x - x.width, axis=1)
    mirror_df['ul_corner_x'] = mirror_df.apply(lambda x: x.ul_corner_x - x.width, axis=1)
    mirror_df['ur_corner_x'] = mirror_df.apply(lambda x: x.ur_corner_x - x.width, axis=1)
    mirror_df['br_corner_x'] = mirror_df.apply(lambda x: x.br_corner_x - x.width, axis=1)
    mirror_df['bl_corner_x'] = mirror_df.apply(lambda x: x.bl_corner_x - x.width, axis=1)
    mirror_df['left_net_x'] = mirror_df.apply(lambda x: x.left_net_x - x.width, axis=1)
    mirror_df['right_net_x'] = mirror_df.apply(lambda x: x.right_net_x - x.width, axis=1)
    mirror_df['player_A_court_x'] = mirror_df.apply(
        lambda x: -1 if x.player_A_court_x ==-1 else x.player_A_court_x - court_width, axis=1)
    mirror_df['player_B_court_x'] = mirror_df.apply(
        lambda x: -1 if x.player_B_court_x ==-1 else x.player_B_court_x - court_width, axis=1)
    mirror_df['player_A_img_x'] = mirror_df.apply(
        lambda x: -1 if x.player_A_img_x ==-1 else x.player_A_img_x - x.width, axis=1)
    mirror_df['player_B_img_x'] = mirror_df.apply(
        lambda x: -1 if x.player_B_img_x ==-1 else x.player_B_img_x - x.width, axis=1)

    full_df = pd.concat((df, mirror_df), ignore_index=True)
    print(full_df.shape)
    return full_df
