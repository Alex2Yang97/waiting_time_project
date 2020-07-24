#-*- coding:utf-8 -*-
# @Time     : 2020-02-15 22:48
# @Author   : Zhirui(Alex) Yang
# @Function :

import pandas as pd
import pymysql
import datetime


def get_df_from_sql(sql):
    # Connect to the MySQL database
    db = pymysql.connect(host='localhost',
                         port=3306,
                         user='root',
                         passwd='980719',
                         db='waiting_time_data')
    df = pd.read_sql(sql = sql, con = db)
    return df


# 把字符串转成datetime
def str_to_Datetime(st):
    dt = datetime.datetime.strptime(st, "%Y-%m-%d %H:%M:%S")
    return dt


# 把datetime 转成字符串
def Datetime_to_str(d):
    str_date = d.strftime('%Y-%m-%d %H:%M:%S')
    return str_date


# 计算时间差
def cal_time_inv(before, now):
    before = str_to_Datetime(before)
    now = str_to_Datetime(now)
    if now >= before:
        inv = (now - before).seconds
        return inv
    else:
        return 0


# 因为部分特征有多个取值，因此对于这样的数据，如果在DataFrame 中变成一个list
# 一个取值的，还是值，而不是list
def get_list(x):
    x = list(x)
    if len(x) == 1:
        return x[0]
    else:
        return x


# 填补缺失值
def fill_nan(data, columns, data_type):
    # 因为需要进行onehot encoding，所以在拼接数据之前，先进行数据格式的处理
    if data_type == 'number':
        for col in columns:
            try:
                data[col].fillna(0, inplace=True)
            except:
                pass
    else:
        for col in columns:
            try:
                data[col].fillna('Unknown', inplace=True)
                data[col] = data[col].astype(str)
            except:
                pass
    return data


# 将分类变量转为category 类型
def cate_type(data, feature_cate):
    for col in feature_cate:
        data[col] = data[col].astype('category')
    return data