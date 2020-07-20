#-*- coding:utf-8 -*-
# @Time     : 2020-07-11 18:21
# @Author   : Zhirui(Alex) Yang
# @Function :

import os
import pymysql
import pandas as pd

from utils.db_conf import WAITING_TIME_CONF


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def read_df_from_sql(sql, db_config=WAITING_TIME_CONF):
    con = pymysql.connect(**db_config)
    df = pd.read_sql(sql=sql, con=con)
    con.close()
    return df


def drop_cols_with_same_values(df):
    for col in df.columns:
        if len(df[col].unique()) == 1:
            df = df.drop(col, axis=1)
    return df

