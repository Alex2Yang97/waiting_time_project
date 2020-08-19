"""
@Time: 2020/8/17 18:08
@Author: Zhirui(Alex) Yang
@E-mail: 1076830028@qq.com
@Program: 
"""


import os
import random

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer

import tensorflow as tf
from tensorflow.keras import layers, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from utils.logger import logger
from data_process import get_apptointment_info, get_treat_info
from wtp.duration.predict_lgb_model import NUM_FEATURES, CATE_FEATURES
from wtp.duration.config_duration import DT_MODEL_DIR


def one_hot_encoding(processed_data):
    cate_onehot_data = pd.DataFrame({})
    update_cate_features = []
    for feature in CATE_FEATURES:
        tmp = pd.get_dummies(processed_data[[feature]], prefix=f"{feature}_")
        update_cate_features.extend(tmp.columns)
        cate_onehot_data = pd.concat([cate_onehot_data, tmp], axis=1)

    cate_onehot_data['AppointmentSerNum'] = processed_data['AppointmentSerNum']
    cate_onehot_data = cate_onehot_data.groupby(by='AppointmentSerNum').sum()
    return cate_onehot_data, update_cate_features


def fill_num(processed_data):
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_mean.fit(processed_data[NUM_FEATURES])
    processed_data.loc[:, NUM_FEATURES] = imp_mean.transform(processed_data[NUM_FEATURES])
    return processed_data


def split_feature_label(all_data, update_cate_features):
    patients_lst = []
    train_samples_lst = []
    label_samples_lst = []
    for pat, sample in all_data.groupby('PatientSerNum'):
        sample = sample[NUM_FEATURES + update_cate_features]
        label_samples_lst.append(sample.iloc[-1, 1])
        sample.iloc[-1, 1] = 0
        patients_lst.append(pat)
        train_samples_lst.append(sample.values)

    return patients_lst, train_samples_lst, label_samples_lst


def process_sequence_data(processed_data):
    logger.debug(f'Fill zero in nan!')
    processed_data = fill_num(processed_data)

    logger.debug(f'Process numerical features!')
    num_features_single_value = ['age', 'Scheduled_duration', 'Actual_duration']
    num_data_single_value = processed_data[num_features_single_value + ['AppointmentSerNum']]
    num_data_single_value = num_data_single_value.groupby(by='AppointmentSerNum').mean()
    num_data_single_value = num_data_single_value.reset_index(drop=False)
    num_features_multiple_value = ['ImagesTaken', 'MU', 'MUCoeff', 'TreatmentTime']
    num_data_multiple_value = processed_data[num_features_multiple_value + ['AppointmentSerNum']]
    num_data_multiple_value = num_data_multiple_value.groupby(by='AppointmentSerNum').sum()
    num_data_multiple_value = num_data_multiple_value.reset_index(drop=False)
    num_data = pd.merge(num_data_single_value, num_data_multiple_value, on='AppointmentSerNum', how='inner')

    logger.debug(f'Encode categorical features!')
    cate_onehot_data, update_cate_features = one_hot_encoding(processed_data)
    feature_data = pd.merge(num_data, cate_onehot_data, on='AppointmentSerNum', how='inner')

    logger.debug(f'Add appointment information!')
    information_features = ['PatientSerNum', 'AppointmentSerNum',
                            'ScheduledStartTime', 'ScheduledEndTime', 'ActualStartDate', 'ActualEndDate']# FractionNumber
    information_data = processed_data[information_features]
    information_data = information_data.drop_duplicates().reset_index(drop=True)
    all_data = pd.merge(feature_data, information_data, on='AppointmentSerNum', how='inner')

    logger.debug(f'Split features and labels!')
    all_data = all_data.sort_values(by=['PatientSerNum', 'ScheduledStartTime']).reset_index(drop=True)
    patients_lst, train_samples_lst, label_samples_lst = split_feature_label(all_data, update_cate_features)

    return patients_lst, train_samples_lst, label_samples_lst


def sequence_model():
    model = Sequential()
    model.add(
        layers.LSTM(128,
                    batch_input_shape=(None, None, 209),
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


def generate_data(x_set, y_set, batch_size):
    i = 0
    while True:
        feature_samples = []
        label_samples = []
        for b in range(batch_size):
            if i == len(x_set):
                i = 0
                random.seed(1)
                random.shuffle(x_set)
                random.seed(1)
                random.shuffle(y_set)

            feature_samples.append(x_set[i])
            label_samples.append(y_set[i])

            i = i + 1

        feature_samples = tf.keras.preprocessing.sequence.pad_sequences(np.array(feature_samples), padding="pre")

        yield feature_samples, np.array(label_samples)
        # yield ({'input': train_samples}, {'output': batch_samples})


def split_train_test(train_samples_lst, label_samples_lst, seed=1):
    random.seed(seed)
    random.shuffle(train_samples_lst)
    random.seed(seed)
    random.shuffle(label_samples_lst)

    data_length = len(train_samples_lst)
    train_set = train_samples_lst[: int(data_length * 0.9)]
    label_train_set = label_samples_lst[: int(data_length * 0.9)]
    val_set = train_samples_lst[int(data_length * 0.9): int(data_length * 0.95)]
    label_val_set = label_samples_lst[int(data_length * 0.9): int(data_length * 0.95)]
    test_set = train_samples_lst[int(data_length * 0.95):]
    label_test_set = label_samples_lst[int(data_length * 0.95):]
    return train_set, label_train_set, val_set, label_val_set, test_set, label_test_set


def train_and_test(train_set, label_train_set, val_set, label_val_set, test_set, label_test_set, seed, model_name):
    logger.debug(f'Start training model for {seed}!')
    model_sequence = sequence_model()
    opt = optimizers.Adam(lr=0.001)
    model_sequence.compile(
        optimizer=opt,
        loss='mae',
        metrics=['mean_absolute_percentage_error', 'mae']
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      min_delta=1e5,
                                                      patience=5,
                                                      verbose=1,
                                                      restore_best_weights=True)
    checkpoint_path_industry = os.path.join(DT_MODEL_DIR, f"{model_name}.h5")
    cp_callback_model = tf.keras.callbacks.ModelCheckpoint(checkpoint_path_industry,
                                                           monitor='val_loss',
                                                           verbose=0,
                                                           save_best_only=True,
                                                           save_weights_only=False,
                                                           mode='min',
                                                           period=1)

    random.seed(seed)
    random.shuffle(train_set)
    random.seed(seed)
    random.shuffle(label_train_set)
    model_sequence.fit_generator(generate_data(train_set, label_train_set, 32),
                                 steps_per_epoch=len(train_set) // 32,
                                 epochs=100,
                                 callbacks=[early_stopping, cp_callback_model],
                                 # batch_size=256,
                                 validation_data=generate_data(val_set, label_val_set, 32),
                                 validation_steps=len(val_set) // 32)

    logger.debug(f'Start testing model!')
    padding_test_set = tf.keras.preprocessing.sequence.pad_sequences(np.array(test_set), padding="pre")
    y_pred = model_sequence.predict(padding_test_set)
    residual = np.array(label_test_set) - y_pred.reshape(-1, )
    logger.debug(f"MAE: {np.mean(np.abs(residual))}")
    logger.debug(f"MAPE: {100. * np.mean(np.abs(residual / np.array(label_test_set)))}")
    return y_pred


if __name__ == '__main__':
    processed_appointment_data = get_apptointment_info()
    processed_treatment_data = get_treat_info()
    processed_data = pd.merge(processed_appointment_data, processed_treatment_data,
                              on=['PatientSerNum', 'date'], how='inner')

    _, train_samples_lst, label_samples_lst = process_sequence_data(processed_data)

    train_set, label_train_set, val_set, label_val_set, test_set, label_test_set = \
        split_train_test(train_samples_lst, label_samples_lst)

    pred_y_ensemble = []
    for seed in range(10):
        pred_y = train_and_test(train_set, label_train_set, val_set, label_val_set, test_set, label_test_set,
                                seed=seed, model_name=f'sequence_model_{seed}.h5')
        pred_y_ensemble.append(pred_y.reshape(-1, ))






