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
from utils.funcs import save_to_pkl, mae, mape

from data_process import get_apptointment_info, get_treat_info

from wtp.duration.config_duration import DT_MODEL_DIR, DT_FIG_DIR, DT_RES_DIR


NUM_FEATURES = ['Scheduled_duration', 'Actual_duration',
                'age',
                'TreatmentTime', 'ImagesTaken',
                'MU', 'MUCoeff']

# RadiationId
CATE_FEATURES = ['dxt_AliasName', 'AliasSerNum',
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


def data_split_by_date(data, ratio=0.9):
    train_data = data.iloc[: int(data.shape[0] * ratio)]
    train_data = train_data.reset_index(drop=True)
    test_data = data.iloc[int(data.shape[0] * ratio):]
    test_data = test_data.reset_index(drop=True)

    return train_data, test_data


def generate_xy(data, label='Actual_duration'):
    y = data[[label]]
    data_ = data.drop([label], axis=1)
    return data_, y


def lgb_reg_train(train_x, train_y, val_x, val_y, model_args=None, train_args=None):
    logger.info('Start training model!')
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


if __name__ == '__main__':
    processed_appointment_data = get_apptointment_info()
    logger.debug(f'process_appointment_data shap {processed_appointment_data.shape}')
    processed_treatment_data = get_treat_info()
    logger.debug(f'process_treatment_data shap {processed_treatment_data.shape}')

    data_display, data, origin_data = select_feature(processed_appointment_data, processed_treatment_data)

    train_data, test_data = data_split_by_date(data)
    train_data, val_data = data_split_by_date(train_data)
    x_train, y_train = generate_xy(train_data)
    x_val, y_val = generate_xy(val_data)
    logger.info('Train model for LightGBM model with scheduled duration!')
    # model_args = {'objective': 'rmse'}
    model = lgb_reg_train(x_train, y_train, x_val, y_val)
    save_to_pkl(model.booster_, f'lgb_mape.pkl', DT_MODEL_DIR)

    no_doc_train_data = train_data.drop(columns=['Scheduled_duration']).copy()
    no_doc_val_data = val_data.drop(columns=['Scheduled_duration']).copy()
    no_doc_test_data = test_data.drop(columns=['Scheduled_duration']).copy()
    no_doc_x_train, no_doc_y_train = generate_xy(no_doc_train_data)
    no_doc_x_val, no_doc_y_val = generate_xy(no_doc_val_data)
    logger.info('Train model for LightGBM model without scheduled duration!')
    model = lgb_reg_train(no_doc_x_train, no_doc_y_train, no_doc_x_val, no_doc_y_val)
    save_to_pkl(model.booster_, f'no_doc_lgb_mape.pkl', DT_MODEL_DIR)

    logger.info('Testing for LightGBM model with scheduled duration!')
    x_test, y_test = generate_xy(test_data)
    y_pred_lgb = model.booster_.predict(x_test) # num_iteration=booster.best_iteration_
    logger.debug(f'The Mean Absolute Error is {mae(y_test, y_pred_lgb)}!')
    logger.debug(f'The Mean Absolute Percentage Error is {mape(y_test, y_pred_lgb)}!')

    logger.info('Testing for LightGBM model without scheduled duration!')
    no_doc_x_test, no_doc_y_test = generate_xy(no_doc_test_data)
    no_doc_y_pred_lgb = model.booster_.predict(no_doc_x_test)  # num_iteration=booster.best_iteration_
    logger.debug(f'The Mean Absolute Error is {mae(no_doc_y_test, no_doc_y_pred_lgb)}!')
    logger.debug(f'The Mean Absolute Percentage Error is {mape(no_doc_y_test, no_doc_y_pred_lgb)}!')

    # 人工基准
    logger.info('Testing for Doctor experience!')
    y_pred_doctor = x_test['Scheduled_duration'].tolist()
    logger.debug(f'The Mean Absolute Error is {mae(y_test, y_pred_doctor)}!')
    logger.debug(f'The Mean Absolute Percentage Error is {mape(y_test, y_pred_doctor)}!')






