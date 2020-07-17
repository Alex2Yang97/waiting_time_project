#-*- coding:utf-8 -*-
# @Time     : 2020-07-11 18:21
# @Author   : Zhirui(Alex) Yang
# @Function :

import os
import pymysql

import pandas as pd


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def get_df_from_sql(sql):
    # Connect to the MySQL database
    db = pymysql.connect(host='localhost',
                         port=3306,
                         user='root',
                         passwd='root',
                         db='originaldata')
    df = pd.read_sql(sql = sql, con = db)
    return df