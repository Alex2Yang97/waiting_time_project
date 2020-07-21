#-*- coding:utf-8 -*-
# @Time     : 2020-07-11 18:21
# @Author   : Zhirui(Alex) Yang
# @Function :

import os
import joblib
import pymysql
import pandas as pd
import numpy as np

from utils.db_conf import WAITING_TIME_CONF


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def read_df_from_sql(sql, db_config=WAITING_TIME_CONF):
    con = pymysql.connect(**db_config)
    df = pd.read_sql(sql=sql, con=con)
    con.close()
    return df


def save_to_pkl(obj, filename, dir_path='./'):
    file_path = os.path.join(dir_path, filename)
    joblib.dump(obj, file_path)


def drop_cols_with_same_values(df):
    for col in df.columns:
        if len(df[col].unique()) == 1:
            df = df.drop(col, axis=1)
    return df


def rmse(y_true, y_pred):
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred have different shape for "
            f"{y_true.shape} != {y_pred.shape}")
    sq = sum((y_pred - y_true)**2) / len(y_pred)
    return np.sqrt(sq)


def mae(y_true, y_pred):
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred have different shape for "
            f"{y_true.shape} != {y_pred.shape}")
    return sum(np.abs((y_pred - y_true))) / len(y_pred)


def mape(y_true, y_pred):
    y_true = np.array(y_true).astype(np.float64)
    y_pred = np.array(y_pred).astype(np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred have different shape for "
            f"{y_true.shape} != {y_pred.shape}")
    return np.nanmean(
        np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-7, None)))