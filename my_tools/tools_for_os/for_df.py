#-*- coding:utf-8 -*-
# @Time     : 2020-02-15 22:48
# @Author   : Zhirui(Alex) Yang
# @Function :

import pandas as pd
import pymysql


def get_df_from_sql(sql):
    # Connect to the MySQL database
    db = pymysql.connect(host='localhost',
                         port=3306,
                         user='root',
                         passwd='root',
                         db='originaldata')
    df = pd.read_sql(sql = sql, con = db)
    return df
