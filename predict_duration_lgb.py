"""
@Time: 2020/7/20 10:17
@Author: Zhirui(Alex) Yang
@E-mail: 1076830028@qq.com
@Program: 
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn import preprocessing

from utils.logger import logger


FEATURE_NUM = ['Scheduled_duration', 'Actual_duration',
               'age',
               'TreatmentTime_total', 'ImagesTaken_total',
               'MU_total', 'MUCoeff_total']

# RadiationId
FEATURE_CATE = ['dxt_AliasName', 'AliasSerNum',
                'Sex',
                'month', 'week', 'hour',
                'DoctorSerNum',
                'TreatmentOrientation',
                'FractionNumber',
                'UserName',
                'CourseId',
                'ResourceSerNum']


def select_feature(processed_appt_data, processed_treat_data):
    logger.debug('Merge processed_appointment_data and processed_treat_data!')
    processed_data = pd.merge(processed_appt_data, processed_treat_data, on=['PatientSerNum', 'date'], how='inner')
    processed_data = processed_data.sort_values(
        by=['PatientSerNum', 'AppointmentSerNum', 'ScheduledStartTime', 'FractionNumber'])

    origin_data = processed_data.copy()

    processed_data = processed_data[FEATURE_NUM + FEATURE_CATE]
    processed_data = processed_data.drop_duplicates()

    # 数值变量暂时不做处理
    logger.debug('Process numerical feature!')
    data_num = processed_data[FEATURE_NUM]
    data_num = data_num.fillna(0)
    # data_num = log1p(data_num)

    # 非数值变量进行categorical encoding
    logger.debug('Process categorical feature!')
    data_cate_display = processed_data[FEATURE_CATE]
    data_cate_display = data_cate_display.fillna('NaN')
    data_cate_display = data_cate_display.astype(str)

    # OrdinalEncoder 处理非数值变量
    feature_encoder = preprocessing.OrdinalEncoder()
    feature_encoder.fit(data_cate_display.values)
    data_cate = feature_encoder.transform(data_cate_display.values)
    data_cate = pd.DataFrame(data_cate, columns=data_cate_display.columns)
    # 还需要换成category 格式，这样lightgbm 会按照类别变量的方式进行处理
    data_cate = data_cate.astype('category')

    logger.debug('Concat data_num and data_cate!')
    data_num = data_num.reset_index(drop=True)
    data_cate_display = data_cate_display.reset_index(drop=True)
    data_display = pd.concat([data_num, data_cate_display], axis=1)
    data_display = data_display.sample(frac=1, random_state=1).reset_index(drop=True)

    data_cate = data_cate.reset_index(drop=True)
    data = pd.concat([data_num, data_cate], axis=1)
    data = data.sample(frac=1, random_state=1).reset_index(drop=True)

    origin_data = origin_data.sample(frac=1, random_state=1).reset_index(drop=True)

    return data_display, data, origin_data


# def train_lgb():
#
#     label = data[['Actual_duration']]
#     data.drop(['Actual_duration'], axis=1, inplace=True)
#
#     train_x = data.iloc[: int(data.shape[0] * 0.9)]
#     train_y = label.iloc[: int(data.shape[0] * 0.9)]
#     print(f'\nThe number of train set is {train_x.shape[0]}')
#
#     test_x = data.iloc[int(data.shape[0] * 0.9):]
#     test_y = label.iloc[int(data.shape[0] * 0.9):]
#     print(f'The number of test set is {test_x.shape[0]}')
#
#     print('Finish getting')


def lgb_reg_train(train_x, train_y, val_x, val_y, model_args, train_args):
    default_model_args = {
        'num_leaves': 64, 'objective': 'mape', 'max_depth': -1,
        'learning_rate': 0.01, 'min_child_samples': 5, 'n_estimators': 100000,
        'subsample': 0.8, 'max_cat_threshold': 400, 'colsample_bytree': 1,
        'subsample_freq': 1, 'n_jobs': 4, 'silent': 1}
    default_train_args = {
        'early_stopping_rounds': 100, 'eval_metric': 'mape', 'verbose': 10}

    if model_args is not None:
        default_model_args.update(model_args)
    if train_args is not None:
        default_train_args.update(train_args)

    lgb_model = lgb.LGBMRegressor(**default_model_args)

    lgb_model.fit(train_x, train_y,
                  eval_set=[(train_x, train_y), (val_x, val_y)],
                  **default_train_args)

    return lgb_model