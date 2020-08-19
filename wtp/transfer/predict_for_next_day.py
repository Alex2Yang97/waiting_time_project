"""
@Time: 2020/8/19 11:30
@Author: Zhirui(Alex) Yang
@E-mail: 1076830028@qq.com
@Program: 
"""

import pymysql
import datetime
import time
# import warnings
# warnings.filterwarnings("ignore")
import random

import pandas as pd
import numpy as np

from utils.funcs import read_df_from_sql, cal_time_inv
from utils.logger import logger
from data_process import get_apptointment_info, get_treat_info
from wtp.duration.predict_sequence_model_v1 import one_hot_enc, process_sequence_data, filter_data
from wtp.duration.predict_lgb_model import lgb_reg_train


ROOM_NUM_FEATURES = ['Scheduled_duration', 'age', 'TreatmentTime_total',
                     'ImagesTaken_total', 'MU_total', 'MUCoeff_total', 'order',
                     'last_Scheduled_duration', 'last_age', 'last_TreatmentTime_total',
                     'last_ImagesTaken_total', 'last_MU_total', 'last_MUCoeff_total', 'last_order',
                     'Interval_number', 'Transfer_duration']

ROOM_CATE_FEATURES = ['dxt_AliasName', 'Sex', 'AliasSerNum', 'hour', 'DoctorSerNum',
                      'TreatmentOrientation', 'FractionNumber', 'UserName', 'CourseId',
                      'last_dxt_AliasName', 'last_Sex', 'last_AliasSerNum', 'last_hour', 'last_DoctorSerNum',
                      'last_TreatmentOrientation', 'last_FractionNumber', 'last_UserName', 'last_CourseId',
                      'ResourceSerNum', 'month', 'week']


def get_room_information():
    logger.debug('Get table of resourceappointment!')
    data_re_appt = read_df_from_sql('SELECT * FROM resourceappointment').drop(columns=['LastUpdated'])

    data_room = data_re_appt.rename(columns={'ResourceSerNum': 'room'})
    data_room = data_room[['AppointmentSerNum', 'room']].drop_duplicates()
    return data_room


def get_room_last_future_patient(room_info_grouped, room_number):
    room = room_info_grouped.get_group(room_number)

    room = room.sort_values(by='ScheduledStartTime')
    room['order'] = [i + 1 for i in range(room.shape[0])]

    room = room.sort_values(by='ActualStartDate').reset_index(drop=True)

    # 去掉最后一位患者的信息，构成上一位患者的DataFrame
    room_last = room.copy()
    room_last = room_last.drop(labels=[room_last.index[-1]], axis=0).reset_index(drop=True)
    room_last.columns = ['last_' + col for col in room_last.columns]

    # 去掉第一位患者的信息，构成当前患者的DataFrame
    room = room.drop(labels=[room.index[0]], axis=0).reset_index(drop=True)

    # 横向拼接数据
    room = pd.concat([room_last, room], axis=1)

    # 计算实际转换时长，也就是得到标签
    room['Transfer_duration'] = room.apply(
        lambda x: cal_time_inv(x.last_ActualEndDate, x.ActualStartDate) / 60, axis=1)

    # 因为可能有缺失的数据，因此把这个作为一个特征
    room['Interval_number'] = room.apply(lambda x: int(x.Transfer_duration / 15), axis=1)

    return room[ROOM_NUM_FEATURES + ROOM_CATE_FEATURES]


def process_room_sample(series_data):
    logger.info('Process room sample!')

    data_room = get_room_information()
    room_series_data = pd.merge(series_data, data_room, on='AppointmentSerNum', how='left')

    room_info_grouped = room_series_data.groupby(['room', 'date'])

    room_count = room_info_grouped.count()
    room_seq = room_count[room_count.PatientSerNum > 1].index.tolist()
    room_one = room_count[room_count.PatientSerNum == 1].index.tolist()
    logger.debug(f'room with multiple patients: {len(room_seq)}!')
    logger.debug(f'room with one patients: {len(room_one)}!')

    # 形成专门针对room 的DataFrame
    logger.info('Splice adjacent patient data!')
    room_sample_all = pd.DataFrame({})
    for i in range(len(room_seq)):
        # if (i + 1) % 100 == 0:
        #     logger.debug(f'No.{i + 1} Room samples')
        room_number = room_seq[i]
        room = get_room_last_future_patient(room_info_grouped, room_number)
        room_sample_all = pd.concat([room_sample_all, room], axis=0)

    room_sample_all = room_sample_all.sample(frac=1, random_state=1)
    return room_sample_all


def generate_sample(room_sample_all, one_hot_dict):
    logger.info('Generate samples!')
    # 对分类变量进行one-hot encoding处理
    logger.info('Process categorical features by one hot encoding!')
    encode_cate = pd.DataFrame({})

    label_encoder_dxt_AliasName = one_hot_dict['label_encoder_dxt_AliasName']
    encode_cate['last_dxt_AliasName'] = room_sample_all['last_dxt_AliasName'].apply(
        lambda x: sum(label_encoder_dxt_AliasName.transform(np.array(x).reshape(-1, 1))))
    train_x = np.vstack(encode_cate.last_dxt_AliasName.tolist())
    encode_cate['dxt_AliasName'] = room_sample_all['dxt_AliasName'].apply(
        lambda x: sum(label_encoder_dxt_AliasName.transform(np.array(x).reshape(-1, 1))))
    train_x = np.hstack((train_x, np.vstack(encode_cate.dxt_AliasName.tolist())))

    label_encoder_Sex = one_hot_dict['label_encoder_Sex']
    encode_cate['last_Sex'] = room_sample_all['last_Sex'].apply(
        lambda x: sum(label_encoder_Sex.transform(np.array(x).reshape(-1, 1))))
    train_x = np.hstack((train_x, np.vstack(encode_cate.last_Sex.tolist())))
    encode_cate['Sex'] = room_sample_all['Sex'].apply(
        lambda x: sum(label_encoder_Sex.transform(np.array(x).reshape(-1, 1))))
    train_x = np.hstack((train_x, np.vstack(encode_cate.Sex.tolist())))

    label_encoder_AliasSerNum = one_hot_dict['label_encoder_AliasSerNum']
    encode_cate['last_AliasSerNum'] = room_sample_all['last_AliasSerNum'].apply(
        lambda x: sum(label_encoder_AliasSerNum.transform(np.array(x).reshape(-1, 1))))
    train_x = np.hstack((train_x, np.vstack(encode_cate.last_AliasSerNum.tolist())))
    encode_cate['AliasSerNum'] = room_sample_all['AliasSerNum'].apply(
        lambda x: sum(label_encoder_AliasSerNum.transform(np.array(x).reshape(-1, 1))))
    train_x = np.hstack((train_x, np.vstack(encode_cate.AliasSerNum.tolist())))

    label_encoder_hour = one_hot_dict['label_encoder_hour']
    encode_cate['last_hour'] = room_sample_all['last_hour'].apply(
        lambda x: sum(label_encoder_hour.transform(np.array(x).reshape(-1, 1))))
    train_x = np.hstack((train_x, np.vstack(encode_cate.last_hour.tolist())))
    encode_cate['hour'] = room_sample_all['hour'].apply(
        lambda x: sum(label_encoder_hour.transform(np.array(x).reshape(-1, 1))))
    train_x = np.hstack((train_x, np.vstack(encode_cate.hour.tolist())))

    label_encoder_DoctorSerNum = one_hot_dict['label_encoder_DoctorSerNum']
    encode_cate['last_DoctorSerNum'] = room_sample_all['last_DoctorSerNum'].apply(
        lambda x: sum(label_encoder_DoctorSerNum.transform(np.array(x).reshape(-1, 1))))
    train_x = np.hstack((train_x, np.vstack(encode_cate.last_DoctorSerNum.tolist())))
    encode_cate['DoctorSerNum'] = room_sample_all['DoctorSerNum'].apply(
        lambda x: sum(label_encoder_DoctorSerNum.transform(np.array(x).reshape(-1, 1))))
    train_x = np.hstack((train_x, np.vstack(encode_cate.DoctorSerNum.tolist())))

    label_encoder_TreatmentOrientation = one_hot_dict['label_encoder_TreatmentOrientation']
    encode_cate['last_TreatmentOrientation'] = room_sample_all['last_TreatmentOrientation'].apply(
        lambda x: sum(label_encoder_TreatmentOrientation.transform(np.array(x).reshape(-1, 1))))
    train_x = np.hstack((train_x, np.vstack(encode_cate.last_TreatmentOrientation.tolist())))
    encode_cate['TreatmentOrientation'] = room_sample_all['TreatmentOrientation'].apply(
        lambda x: sum(label_encoder_TreatmentOrientation.transform(np.array(x).reshape(-1, 1))))
    train_x = np.hstack((train_x, np.vstack(encode_cate.TreatmentOrientation.tolist())))

    label_encoder_FractionNumber = one_hot_dict['label_encoder_FractionNumber']
    encode_cate['last_FractionNumber'] = room_sample_all['last_FractionNumber'].apply(
        lambda x: sum(label_encoder_FractionNumber.transform(np.array(x).reshape(-1, 1))))
    train_x = np.hstack((train_x, np.vstack(encode_cate.last_FractionNumber.tolist())))
    encode_cate['FractionNumber'] = room_sample_all['FractionNumber'].apply(
        lambda x: sum(label_encoder_FractionNumber.transform(np.array(x).reshape(-1, 1))))
    train_x = np.hstack((train_x, np.vstack(encode_cate.FractionNumber.tolist())))

    label_encoder_UserName = one_hot_dict['label_encoder_UserName']
    encode_cate['last_UserName'] = room_sample_all['last_UserName'].apply(
        lambda x: sum(label_encoder_UserName.transform(np.array(x).reshape(-1, 1))))
    train_x = np.hstack((train_x, np.vstack(encode_cate.last_UserName.tolist())))
    encode_cate['UserName'] = room_sample_all['UserName'].apply(
        lambda x: sum(label_encoder_UserName.transform(np.array(x).reshape(-1, 1))))
    train_x = np.hstack((train_x, np.vstack(encode_cate.UserName.tolist())))

    label_encoder_CourseId = one_hot_dict['label_encoder_CourseId']
    encode_cate['last_CourseId'] = room_sample_all['last_CourseId'].apply(
        lambda x: sum(label_encoder_CourseId.transform(np.array(x).reshape(-1, 1))))
    train_x = np.hstack((train_x, np.vstack(encode_cate.last_CourseId.tolist())))
    encode_cate['CourseId'] = room_sample_all['CourseId'].apply(
        lambda x: sum(label_encoder_CourseId.transform(np.array(x).reshape(-1, 1))))
    train_x = np.hstack((train_x, np.vstack(encode_cate.CourseId.tolist())))

    label_encoder_ResourceSerNum = one_hot_dict['label_encoder_ResourceSerNum']
    encode_cate['ResourceSerNum'] = room_sample_all['ResourceSerNum'].apply(
        lambda x: sum(label_encoder_ResourceSerNum.transform(np.array(str(x)).reshape(-1, 1))))
    train_x = np.hstack((train_x, np.vstack(encode_cate.ResourceSerNum.tolist())))

    label_encoder_month = one_hot_dict['label_encoder_month']
    encode_cate['month'] = room_sample_all['month'].apply(
        lambda x: sum(label_encoder_month.transform(np.array(x).reshape(-1, 1))))
    train_x = np.hstack((train_x, np.vstack(encode_cate.month.tolist())))

    label_encoder_week = one_hot_dict['label_encoder_week']
    encode_cate['week'] = room_sample_all['week'].apply(
        lambda x: sum(label_encoder_week.transform(np.array(x).reshape(-1, 1))))
    train_x = np.hstack((train_x, np.vstack(encode_cate.week.tolist())))

    # encode_cate['AliasExpressionSerNum'] = room['AliasExpressionSerNum'].apply(
    #    lambda x: sum(label_encoder_AliasExpressionSerNum.transform(np.array(x).reshape(-1,1))))
    # train_x = np.hstack((train_x, np.vstack(encode_cate.AliasExpressionSerNum.tolist())))

    train_num = room_sample_all[ROOM_NUM_FEATURES]
    train_num.drop('Transfer_duration', axis=1, inplace=True)
    train_x = np.hstack((train_x, train_num))

    train_y = room_sample_all[['Transfer_duration']]

    return train_x, train_y


def split_train_test(x, y):
    samples_length = len(y)
    train_x = x_samples[: int(samples_length * 0.9)]
    train_y = y_labels[: int(samples_length * 0.9)]
    logger.debug(f'The shape of train_x is {train_x.shape}')
    val_x = x_samples[int(samples_length * 0.9): int(samples_length * 0.95)]
    val_y = y_labels[int(samples_length * 0.9): int(samples_length * 0.95)]
    logger.debug(f'The shape of val_x is {val_x.shape}')
    test_x = x_samples[int(samples_length * 0.95):]
    test_y = y_labels[int(samples_length * 0.95):]
    logger.debug(f'The shape of test_x is {test_x.shape}')
    return train_x, train_y, val_x, val_y, test_x, test_y


if __name__ == '__main__':
    processed_appointment_data = get_apptointment_info()
    processed_treatment_data = get_treat_info()
    processed_data = pd.merge(processed_appointment_data, processed_treatment_data,
                              on=['PatientSerNum', 'date'], how='inner')
    processed_seq_data = process_sequence_data(processed_data)
    one_hot_dict = {
        'label_encoder_dxt_AliasName': one_hot_enc('dxt_AliasName', processed_seq_data),
        'label_encoder_Sex': one_hot_enc('Sex', processed_seq_data),
        'label_encoder_AliasSerNum': one_hot_enc('AliasSerNum', processed_seq_data),
        'label_encoder_month': one_hot_enc('month', processed_seq_data),
        'label_encoder_week': one_hot_enc('week', processed_seq_data),
        'label_encoder_hour': one_hot_enc('hour', processed_seq_data),
        'label_encoder_DoctorSerNum': one_hot_enc('DoctorSerNum', processed_seq_data),
        'label_encoder_TreatmentOrientation': one_hot_enc('TreatmentOrientation', processed_seq_data),
        'label_encoder_FractionNumber': one_hot_enc('FractionNumber', processed_seq_data),
        'label_encoder_UserName': one_hot_enc('UserName', processed_seq_data),
        'label_encoder_CourseId': one_hot_enc('CourseId', processed_seq_data),
        'label_encoder_ResourceSerNum': one_hot_enc('ResourceSerNum', processed_seq_data)
    }
    _, _, series_data = filter_data(processed_seq_data)
    room_samples = process_room_sample(series_data)
    x_samples, y_labels = generate_sample(room_samples, one_hot_dict=one_hot_dict)

    train_x, train_y, val_x, val_y, test_x, test_y = split_train_test(x_samples, y_labels)

    # 创建模型，训练模型
    transfer_model = lgb_reg_train(train_x, train_y, val_x, val_y)

    # 测试
    logger.info('Start testing model!')
    pred_y = transfer_model.predict(test_x)
    residual = test_y.values.reshape(-1,) - pred_y
    logger.debug(f"MAE: {np.mean(np.abs(residual))}")
    logger.debug(f"MAPE: {100. * np.mean(np.abs(residual / np.array(pred_y)))}")



