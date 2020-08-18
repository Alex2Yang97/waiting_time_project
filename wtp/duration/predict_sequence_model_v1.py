"""
@Time: 2020/8/13 17:06
@Author: Zhirui(Alex) Yang
@E-mail: 1076830028@qq.com
@Program: 
"""

import random

import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.impute import SimpleImputer

import tensorflow as tf
from tensorflow.keras import layers, optimizers
from tensorflow.keras.models import Sequential
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from utils.logger import logger
from data_process import get_apptointment_info, get_treat_info, get_list
from wtp.duration.predict_lgb_model import FEATURE_NUM, FEATURE_CATE


NN_NUM_FEATURES = ['Scheduled_duration', 'Actual_duration',
                   'age', 'TreatmentTime_total', 'ImagesTaken_total',
                   'MU_total', 'MUCoeff_total', 'Interval_scheduled']

# RadiationId
NN_CATE_FEATURES = ['dxt_AliasName', 'Sex', 'AliasSerNum',
                    'month', 'week', 'hour', 'DoctorSerNum',
                    'TreatmentOrientation', 'FractionNumber',
                    'UserName', 'CourseId', 'ResourceSerNum']


def combine_mul_row_appt(data_part1):
    logger.info('Combine Appointment with multiple rows!')

    new_appt = pd.DataFrame({})
    logger.info('Start processing categorical features!')
    logger.debug('PatientSerNum')
    new_appt['PatientSerNum'] = data_part1.groupby('AppointmentSerNum').PatientSerNum.apply(set)
    new_appt['PatientSerNum'] = new_appt['PatientSerNum'].apply(lambda x: get_list(x))

    logger.debug('Sex')
    new_appt['Sex'] = data_part1.groupby('AppointmentSerNum').Sex.apply(set)
    new_appt['Sex'] = new_appt['Sex'].apply(lambda x: get_list(x))

    logger.debug('DoctorSerNum')
    new_appt['DoctorSerNum'] = data_part1.groupby('AppointmentSerNum').DoctorSerNum.apply(set)
    new_appt['DoctorSerNum'] = new_appt['DoctorSerNum'].apply(lambda x: get_list(x))

    logger.debug('date')
    new_appt['date'] = data_part1.groupby('AppointmentSerNum').date.apply(set)
    new_appt['date'] = new_appt['date'].apply(lambda x: get_list(x))

    logger.debug('ScheduledStartTime')
    new_appt['ScheduledStartTime'] = data_part1.groupby('AppointmentSerNum').ScheduledStartTime.apply(set)
    new_appt['ScheduledStartTime'] = new_appt['ScheduledStartTime'].apply(lambda x: get_list(x))

    logger.debug('ScheduledEndTime')
    new_appt['ScheduledEndTime'] = data_part1.groupby('AppointmentSerNum').ScheduledEndTime.apply(set)
    new_appt['ScheduledEndTime'] = new_appt['ScheduledEndTime'].apply(lambda x: get_list(x))

    logger.debug('ActualStartDate')
    new_appt['ActualStartDate'] = data_part1.groupby('AppointmentSerNum').ActualStartDate.apply(set)
    new_appt['ActualStartDate'] = new_appt['ActualStartDate'].apply(lambda x: get_list(x))

    logger.debug('ActualEndDate')
    new_appt['ActualEndDate'] = data_part1.groupby('AppointmentSerNum').ActualEndDate.apply(set)
    new_appt['ActualEndDate'] = new_appt['ActualEndDate'].apply(lambda x: get_list(x))

    logger.debug('dxt_AliasName')
    new_appt['dxt_AliasName'] = data_part1.groupby('AppointmentSerNum').dxt_AliasName.apply(set)
    new_appt['dxt_AliasName'] = new_appt['dxt_AliasName'].apply(lambda x: get_list(x))

    logger.debug('AliasSerNum')
    new_appt['AliasSerNum'] = data_part1.groupby('AppointmentSerNum').AliasSerNum.apply(set)
    new_appt['AliasSerNum'] = new_appt['AliasSerNum'].apply(lambda x: get_list(x))

    logger.debug('CourseSerNum')
    new_appt['CourseSerNum'] = data_part1.groupby('AppointmentSerNum').CourseSerNum.apply(set)
    new_appt['CourseSerNum'] = new_appt['CourseSerNum'].apply(lambda x: get_list(x))

    logger.debug('PlanSerNum')
    new_appt['PlanSerNum'] = data_part1.groupby('AppointmentSerNum').PlanSerNum.apply(set)
    new_appt['PlanSerNum'] = new_appt['PlanSerNum'].apply(lambda x: get_list(x))

    logger.debug('TreatmentOrientation')
    new_appt['TreatmentOrientation'] = data_part1.groupby('AppointmentSerNum').TreatmentOrientation.apply(set)
    new_appt['TreatmentOrientation'] = new_appt['TreatmentOrientation'].apply(lambda x: get_list(x))

    logger.debug('month')
    new_appt['month'] = data_part1.groupby('AppointmentSerNum').month.apply(set)
    new_appt['month'] = new_appt['month'].apply(lambda x: get_list(x))

    logger.debug('week')
    new_appt['week'] = data_part1.groupby('AppointmentSerNum').week.apply(set)
    new_appt['week'] = new_appt['week'].apply(lambda x: get_list(x))

    logger.debug('hour')
    new_appt['hour'] = data_part1.groupby('AppointmentSerNum').hour.apply(set)
    new_appt['hour'] = new_appt['hour'].apply(lambda x: get_list(x))

    logger.debug('AppointmentSerNum')
    new_appt['AppointmentSerNum'] = new_appt.index.tolist()

    logger.info('Start processing numerical features!')
    logger.debug('age')
    new_appt['age'] = data_part1.groupby('AppointmentSerNum').age.mean()

    logger.debug('Scheduled_duration')
    new_appt['Scheduled_duration'] = data_part1.groupby('AppointmentSerNum').Scheduled_duration.mean()

    logger.debug('Actual_duration')
    new_appt['Actual_duration'] = data_part1.groupby('AppointmentSerNum').Actual_duration.mean()

    new_appt = new_appt.reset_index(drop=True)
    return new_appt


def combine_mul_row_treat(data_part2):
    logger.info('Combine Treatment with multiple rows!')

    new_treat = pd.DataFrame({})
    logger.info('Start processing categorical features!')
    logger.debug('FractionNumber')
    new_treat['FractionNumber'] = data_part2.groupby(['PatientSerNum', 'date']).FractionNumber.apply(set)
    new_treat['FractionNumber'] = new_treat['FractionNumber'].apply(lambda x: get_list(x))

    logger.debug('UserName')
    new_treat['UserName'] = data_part2.groupby(['PatientSerNum', 'date']).UserName.apply(set)
    new_treat['UserName'] = new_treat['UserName'].apply(lambda x: get_list(x))

    logger.debug('RadiationSerNum')
    new_treat['RadiationSerNum'] = data_part2.groupby(['PatientSerNum', 'date']).RadiationSerNum.apply(set)
    new_treat['RadiationSerNum'] = new_treat['RadiationSerNum'].apply(lambda x: get_list(x))

    #     print('Start RadiationId')
    #     new_treat['RadiationId'] = data_part2.groupby(['PatientSerNum', 'date']).RadiationId.apply(set)
    #     new_treat['RadiationId'] = new_treat['RadiationId'].apply(lambda x: get_list(x))

    logger.debug('ResourceSerNum')
    new_treat['ResourceSerNum'] = data_part2.groupby(['PatientSerNum', 'date']).ResourceSerNum.apply(set)
    new_treat['ResourceSerNum'] = new_treat['ResourceSerNum'].apply(lambda x: get_list(x))

    logger.debug('CourseId')
    new_treat['CourseId'] = data_part2.groupby(['PatientSerNum', 'date']).CourseId.apply(set)
    new_treat['CourseId'] = new_treat['CourseId'].apply(lambda x: get_list(x))

    logger.debug('PatientSerNum')
    new_treat['PatientSerNum'] = new_treat.index.get_level_values(level=0).tolist()

    logger.debug('date')
    new_treat['date'] = new_treat.index.get_level_values(level=1).tolist()

    logger.info('Start processing numerical features!')
    logger.debug('ImagesTaken_total')
    new_treat['ImagesTaken_total'] = data_part2.groupby(['PatientSerNum', 'date']).ImagesTaken.sum()

    logger.debug('MU_total')
    new_treat['MU_total'] = data_part2.groupby(['PatientSerNum', 'date']).MU.sum()

    logger.debug('MUCoeff_total')
    new_treat['MUCoeff_total'] = data_part2.groupby(['PatientSerNum', 'date']).MUCoeff.sum()

    logger.debug('TreatmentTime_total')
    new_treat['TreatmentTime_total'] = data_part2.groupby(['PatientSerNum', 'date']).TreatmentTime.sum()

    new_treat = new_treat.reset_index(drop=True)
    return new_treat


# 数字也能够正常处理，比如出现0，同样会正常处理，并不会认为0 是没有值
def one_hot_enc(feature, data):
    one_hot_encoder = preprocessing.OneHotEncoder(handle_unknown='ignore', sparse=False)
    one_hot_encoder.fit(data[[feature]])
    return one_hot_encoder


def process_sequence_data(processed_data_):
    logger.debug('Process sequence data!')
    processed_data = processed_data_.copy()
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_mean.fit(processed_data[FEATURE_NUM])
    processed_data.loc[:, FEATURE_NUM] = imp_mean.transform(processed_data[FEATURE_NUM])

    cate_features = processed_data[FEATURE_CATE].select_dtypes(include=['category', 'object']).columns
    cate_features_number = [i for i in FEATURE_CATE if i not in cate_features]

    processed_data.loc[:, cate_features] = processed_data.loc[:, cate_features].fillna('NULL').reset_index(drop=True)
    processed_data.loc[:, cate_features_number] = processed_data.loc[:, cate_features_number].fillna(0).reset_index(
        drop=True)

    for col in processed_data.columns:
        if col in FEATURE_CATE:
            processed_data[col] = processed_data[col].astype('category').reset_index(drop=True)
        if col in FEATURE_NUM:
            processed_data[col] = processed_data[col].astype('float').reset_index(drop=True)

    return processed_data


def filter_data(processed_data):
    new_appt = combine_mul_row_appt(processed_data)
    new_treat = combine_mul_row_treat(processed_data)
    data_with_multi = pd.merge(new_appt, new_treat, on=['PatientSerNum', 'date'], how='inner')
    data_with_multi = data_with_multi.sort_values(
        by=['PatientSerNum', 'AppointmentSerNum', 'ScheduledStartTime']).reset_index(drop=True)
    data_with_multi = data_with_multi[(data_with_multi.Actual_duration >= 10)
                                      & (data_with_multi.Actual_duration <= 60)].reset_index(drop=True)
    logger.debug('Select Patient with one more Appointment')
    data_with_multi_grouped = data_with_multi.groupby('PatientSerNum')
    series_count = data_with_multi_grouped.count()
    series_seq = series_count[series_count.AppointmentSerNum > 1].index.tolist()
    # series_one = series_count[series_count.AppointmentSerNum == 1].index.tolist()
    series_seq = pd.DataFrame({'PatientSerNum': series_seq})
    series_data = pd.merge(series_seq, data_with_multi, on='PatientSerNum', how='inner').sort_values(
        by=['PatientSerNum', 'ScheduledStartTime']).reset_index(drop=True)
    series_data_grouped = series_data.groupby('PatientSerNum')
    pat_lst = shuffle(list(series_data_grouped.groups.keys()), random_state=1)
    logger.debug(f'The total number of patient is {len(pat_lst)}')
    return pat_lst, series_data_grouped


def sequence_model():
    model = Sequential()
    model.add(
        layers.LSTM(128,
                    batch_input_shape=(None, None, 299),
                    dropout=0.1,
                    recurrent_dropout=0.5,
                    name='input'
                    )
    )
    # model_sequence.add(layers.LSTM(
    #         output_dim = 32,
    #         ))
    # stateful = True 本次batch的参数返回到下一次的训练中
    model.add(layers.Dense(32))
    model.add(layers.Dense(1))
    return model


def split_train_test(pat_lst, seed=1):
    random.seed(seed)
    random.shuffle(pat_lst)

    data_length = len(pat_lst)
    train_lst = pat_lst[: int(0.9 * data_length)]
    val_lst = pat_lst[int(0.9 * data_length): int(0.95 * data_length)]
    test_lst = pat_lst[int(0.95 * data_length):]
    logger.debug(f'The number of train_list {len(train_lst)}')
    logger.debug(f'The number of train_list {len(val_lst)}')
    logger.debug(f'The number of train_list {len(test_lst)}')
    return train_lst, val_lst, test_lst


def generate_sample(train_sample_,
                    label_encoder_dxt_AliasName, label_encoder_Sex, label_encoder_AliasSerNum,
                    label_encoder_month, label_encoder_week, label_encoder_hour,
                    label_encoder_DoctorSerNum, label_encoder_FractionNumber, label_encoder_UserName,
                    label_encoder_CourseId, label_encoder_ResourceSerNum):

    train_sample = train_sample_.reset_index(drop=True)

    # 样本标签
    train_y = np.array([train_sample.Actual_duration.iloc[-1]])

    # 最后一个appointment 为我们需要预测的真实治疗时长所对应的appointment，所以需要设置为0
    train_sample.Actual_duration.iloc[-1] = 0

    # 因为存在相隔很远的两次预约，因此，构造特征Interval_scheduled 来度量两次预约之间的距离
    train_sample['Last_ScheduledStartTime'] = train_sample.ScheduledStartTime.shift(
        periods=1, fill_value=train_sample.ScheduledStartTime.iloc[0])
    train_sample['Interval_scheduled'] = train_sample.apply(
        lambda x: (x.ScheduledStartTime - x.Last_ScheduledStartTime).days, axis=1)

    # 对分类变量进行one-hot encoding处理
    encode_cate = pd.DataFrame({})

    # 这个地方将Sex 作为序列的一部分，并不是在最后的隐藏层进行合并
    encode_cate['Sex'] = train_sample['Sex'].apply(
        lambda x: sum(label_encoder_Sex.transform(np.array(x).reshape(-1, 1))))
    train_x = np.vstack(encode_cate.Sex.tolist())

    encode_cate['dxt_AliasName'] = train_sample['dxt_AliasName'].apply(
        lambda x: sum(label_encoder_dxt_AliasName.transform(np.array(x).reshape(-1, 1))))
    train_x = np.hstack((train_x, np.vstack(encode_cate.dxt_AliasName.tolist())))

    train_sample['AliasSerNum'] = train_sample['AliasSerNum'].astype(str)
    encode_cate['AliasSerNum'] = train_sample['AliasSerNum'].apply(
        lambda x: sum(label_encoder_AliasSerNum.transform(np.array(x).reshape(-1, 1))))
    train_x = np.hstack((train_x, np.vstack(encode_cate.AliasSerNum.tolist())))

    encode_cate['month'] = train_sample['month'].apply(
        lambda x: sum(label_encoder_month.transform(np.array(x).reshape(-1, 1))))
    train_x = np.hstack((train_x, np.vstack(encode_cate.month.tolist())))

    encode_cate['week'] = train_sample['week'].apply(
        lambda x: sum(label_encoder_week.transform(np.array(x).reshape(-1, 1))))
    train_x = np.hstack((train_x, np.vstack(encode_cate.week.tolist())))

    encode_cate['hour'] = train_sample['hour'].apply(
        lambda x: sum(label_encoder_hour.transform(np.array(x).reshape(-1, 1))))
    train_x = np.hstack((train_x, np.vstack(encode_cate.hour.tolist())))

    encode_cate['DoctorSerNum'] = train_sample['DoctorSerNum'].apply(
        lambda x: sum(label_encoder_DoctorSerNum.transform(np.array(x).reshape(-1, 1))))
    train_x = np.hstack((train_x, np.vstack(encode_cate.DoctorSerNum.tolist())))

    encode_cate['TreatmentOrientation'] = train_sample['TreatmentOrientation'].apply(
        lambda x: sum(label_encoder_TreatmentOrientation.transform(np.array(x).reshape(-1, 1))))
    train_x = np.hstack((train_x, np.vstack(encode_cate.TreatmentOrientation.tolist())))

    encode_cate['FractionNumber'] = train_sample['FractionNumber'].apply(
        lambda x: sum(label_encoder_FractionNumber.transform(np.array(x).reshape(-1, 1))))
    train_x = np.hstack((train_x, np.vstack(encode_cate.FractionNumber.tolist())))

    encode_cate['UserName'] = train_sample['UserName'].apply(
        lambda x: sum(label_encoder_UserName.transform(np.array(x).reshape(-1, 1))))
    train_x = np.hstack((train_x, np.vstack(encode_cate.UserName.tolist())))

    encode_cate['CourseId'] = train_sample['CourseId'].apply(
        lambda x: sum(label_encoder_CourseId.transform(np.array(x).reshape(-1, 1))))
    train_x = np.hstack((train_x, np.vstack(encode_cate.CourseId.tolist())))

    encode_cate['ResourceSerNum'] = train_sample['ResourceSerNum'].apply(
        lambda x: sum(label_encoder_ResourceSerNum.transform(np.array(x).reshape(-1, 1))))
    train_x = np.hstack((train_x, np.vstack(encode_cate.ResourceSerNum.tolist())))

    # 将数值变量和非数值变量进行合并
    train_num = train_sample[NN_NUM_FEATURES]
    train_x = np.hstack((train_x, train_num))

    train_x[np.isnan(train_x)] = 0
    return train_x, train_y


def train_and_test(train_lst, val_lst, test_lst, seed, model_name):
    logger.debug('Start training model!')
    model_sequence = sequence_model()
    opt = optimizers.Adam(lr=0.001)
    model_sequence.compile(
        optimizer=opt,
        loss='mae'
    )

    random.seed(seed)
    random.shuffle(train_lst)

    loss_all_lst = []
    loss_batch_lst = []
    val_batch_lst = []

    train_loss_batch = 0
    for i in range(len(train_lst)):
        pat = train_lst[i]
        train_sample = series_data_grouped.get_group(pat)
        train_x_i, train_y_i = generate_sample(train_sample)
        train_x = np.array([train_x_i])
        train_y = np.array([train_y_i])
        train_loss = model_sequence.train_on_batch(train_x, train_y)
        loss_all_lst.append(train_loss)

        # loss 收敛很快，保存每50批次的loss，画出训练的曲线
        if (i + 1) % 50 != 0:
            train_loss_batch = train_loss_batch + train_loss
        else:
            loss_batch_lst.append(train_loss_batch / 50)
            logger.debug(f'Batch {(i + 1) / 50} - train loss: {(i + 1) / 50} {train_loss_batch / 50}')
            train_loss_batch = 0

            val_loss_lst = []
            for j in range(len(val_lst)):
                pat = val_lst[j]
                train_sample = series_data_grouped.get_group(pat)
                train_x_i, train_y_i = generate_sample(train_sample)
                train_x = np.array([train_x_i])
                train_y = np.array([train_y_i])
                val_mae = model_sequence.evaluate(train_x, train_y, verbose=0)
                val_loss_lst.append(val_mae)

            logger.debug(f'Batch {(i + 1) / 50} - validation loss: {sum(val_loss_lst) / len(val_loss_lst)}')
            val_batch_lst.append(val_loss_lst)


if __name__ == '__main__':
    processed_appointment_data = get_apptointment_info()
    processed_treatment_data = get_treat_info()
    processed_data = pd.merge(processed_appointment_data, processed_treatment_data,
                              on=['PatientSerNum', 'date'], how='inner')

    processed_data = process_sequence_data(processed_data)

    label_encoder_dxt_AliasName = one_hot_enc('dxt_AliasName', processed_data)
    label_encoder_Sex = one_hot_enc('Sex', processed_data)
    label_encoder_AliasSerNum = one_hot_enc('AliasSerNum', processed_data)
    label_encoder_month = one_hot_enc('month', processed_data)
    label_encoder_week = one_hot_enc('week', processed_data)
    label_encoder_hour = one_hot_enc('hour', processed_data)
    label_encoder_DoctorSerNum = one_hot_enc('DoctorSerNum', processed_data)
    label_encoder_TreatmentOrientation = one_hot_enc('TreatmentOrientation', processed_data)
    label_encoder_FractionNumber = one_hot_enc('FractionNumber', processed_data)
    label_encoder_UserName = one_hot_enc('UserName', processed_data)
    label_encoder_CourseId = one_hot_enc('CourseId', processed_data)
    label_encoder_ResourceSerNum = one_hot_enc('ResourceSerNum', processed_data)

    pat_lst, series_data_grouped = filter_data(processed_data)
    train_lst, val_lst, test_lst = split_train_test(pat_lst)

    for seed in range(10):
        train_and_test(train_lst, val_lst, test_lst, seed=seed, model_name=f'v1_sequence_model_{seed}.h5')



