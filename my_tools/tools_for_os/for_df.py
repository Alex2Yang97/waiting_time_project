#-*- coding:utf-8 -*-
# @Time     : 2020-02-15 22:48
# @Author   : Zhirui(Alex) Yang
# @Function :

import pandas as pd


def df_set_and_count(df, feature_name):
    value_count = pd.DataFrame({'value': list(df[feature_name].value_counts().index),
                               'count': df[feature_name].value_counts().tolist()})
    return value_count