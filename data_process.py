#-*- coding:utf-8 -*-
# @Time     : 2020-07-11 18:44
# @Author   : Zhirui(Alex) Yang
# @Function :

import gc
import os
import datetime

import pandas as pd
from utils.logger import logger
from utils.funcs import read_df_from_sql
from utils.db_conf import WAITING_TIME_CONF

from utils.config import PROCESSED_DATA_DIR


def get_appt_info():
    logger.info('Get information about appointment!')
    logger.debug('Get table of appointment!')
    data_appointment = read_df_from_sql('SELECT * FROM appointment').drop(columns=['LastUpdated'])
    data_appointment = data_appointment[
        ((data_appointment.AliasSerNum == 31)
         | (data_appointment.AliasSerNum == 23))
        & (data_appointment.ActualStartDate != datetime.datetime(1970, 1, 1, 0, 0))
        & (data_appointment.ActualEndDate != datetime.datetime(1970, 1, 1, 0, 0))
        & (data_appointment.ActualStartDate != data_appointment.ActualEndDate)
        & ((data_appointment.Status == 'Completed')
           | (data_appointment.Status == 'Pt. CompltFinish'))
        & (data_appointment.State == 'Active')]

    logger.debug('Get table of patient!')
    data_pt = read_df_from_sql('SELECT * FROM patient').drop(columns=['LastUpdated'])
    data_ptc = read_df_from_sql('SELECT * FROM patientcopy').drop(columns=['LastUpdated'])

    logger.debug('Merge appointment and patient!')
    merge_data = pd.merge(data_appointment, data_pt, on='PatientSerNum', how='inner')
    merge_data = pd.merge(merge_data, data_ptc, on='PatientSerNum', how='inner')
    del data_appointment, data_pt, data_ptc
    gc.collect()

    logger.debug('Get table of patientdoctor!')
    data_pd = read_df_from_sql('SELECT * FROM patientdoctor').drop(columns=['LastUpdated'])
    data_pd = data_pd[(data_pd.OncologistFlag == 1) & (data_pd.PrimaryFlag == 1)]

    logger.debug('Merge patientdoctor!')
    # data_pd, data_appt AliasSerNum 不同，data_appt_ 中只保留了23、31，data_pd_ 中只有37
    data_pd = data_pd.drop(columns=['AliasSerNum'])
    merge_data = pd.merge(merge_data, data_pd, on='PatientSerNum', how='left')

    logger.debug('Get table of diagnosis!')
    data_dx = read_df_from_sql('SELECT * FROM diagnosis').drop(columns=['LastUpdated'])
    logger.debug('Get table of diagnosis!')
    data_dxt = read_df_from_sql('SELECT * FROM diagnosistranslation').drop(columns=['LastUpdated'])

    logger.debug('Merge diagnosis and diagnosistranslation!')
    data_dxt = data_dxt.rename(columns={'AliasName': 'dxt_AliasName'})
    dx_dxt = pd.merge(data_dx, data_dxt, on='DiagnosisCode', how='left')
    del data_dx, data_dxt
    gc.collect()

    dx_dxt = dx_dxt.drop(columns=['AliasSerNum'])
    dx_dxt = dx_dxt.drop(columns=['PatientSerNum'])
    merge_data = pd.merge(merge_data, dx_dxt, on=['DiagnosisSerNum'], how='inner')

    logger.debug('Get table of course!')
    data_co = read_df_from_sql('SELECT * FROM course')

    logger.debug('Get table of plan!')
    data_pl = read_df_from_sql('SELECT * FROM plan')



    # 拼接pt_appt data_pd
    print(f'\nMerge data_pd_ and pt_appt')

    print(f'\npt_pd_appt shape is {pt_pd_appt.shape}')
    del pt_appt, data_pd
    gc.collect()



    # 拼接pt_pd_appt 和dx_dxt
    print(f'\nMerge pt_pd_appt and dx_dxt')



    # Plan 的预处理
    print('\nProcess data_pl')
    data_pl = data_pl[data_pl.Status == 'TreatApproval']
    data_pl.TreatmentOrientation = data_pl.TreatmentOrientation.apply(
        lambda x: 'NULL' if x == '' else x)
    print(f'The shape of data_pl {data_pl.shape}')

    # 拼接data_pl_ 和data_co
    print(f'\nMerge data_pl and data_co')
    data_pl.drop('AliasSerNum', axis=1, inplace=True)
    data_co.drop('AliasSerNum', axis=1, inplace=True)
    pl_co = pd.merge(data_pl, data_co, on='CourseSerNum', how='inner')
    print(f'pl_co shape is {pl_co.shape}')
    del data_pl, data_co
    gc.collect()

    print(f'\nMerge pt_pd_appt_dx_dxt and pl_co')
    pl_co.drop('AliasExpressionSerNum', axis=1, inplace=True)
    pl_co.drop('DiagnosisSerNum', axis=1, inplace=True)
    pl_co.drop('PrioritySerNum', axis=1, inplace=True)
    pl_co.rename(columns={'Status': 'Status_plan'}, inplace=True)
    data_appt_info = pd.merge(pt_pd_appt_dx_dxt, pl_co,
                              on=['PatientSerNum'],
                              how='inner')
    print(f'\ndata_appt_info shape is {data_appt_info.shape}')
    del pt_pd_appt_dx_dxt, pl_co
    gc.collect()

    print('\nFinish getting data')

    # 　删除值完全相同的列
    print('=' * 20)
    print(f'Drop columns with same values')
    for col in data_appt_info.columns:
        if len(data_appt_info[col].unique()) == 1:
            data_appt_info.drop(col, axis=1, inplace=True)
    print(f'\ndata_appt_info shape {data_appt_info.shape}')

    print('\nFinish dropping')

    print('=' * 20)
    print(f'Create features')

    feature_columns = [
        'dxt_AliasName',
        'DateOfBirth', 'Sex',
        'AliasSerNum', 'PatientSerNum', 'AppointmentSerNum',
        'ScheduledStartTime', 'ScheduledEndTime', 'ActualStartDate', 'ActualEndDate',
        'CourseSerNum',
        'DoctorSerNum',
        'PlanSerNum',
        'TreatmentOrientation',
    ]

    data_appt_info = data_appt_info[feature_columns]

    print(f'\n Patients age')
    # 患者年龄
    data_appt_info['age'] = data_appt_info.apply(lambda x: int((x.ActualStartDate - x.DateOfBirth).days / 365), axis=1)

    print(f'\n Month Date Week Hour')
    # 时间相关的特征，月-日-周-小时
    data_appt_info['month'] = data_appt_info.apply(lambda x: x.ScheduledStartTime.strftime("%m"), axis=1)
    data_appt_info['date'] = data_appt_info.apply(lambda x: x.ScheduledStartTime.strftime("%Y--%m--%d"), axis=1)
    data_appt_info['week'] = data_appt_info.apply(lambda x: x.ScheduledStartTime.strftime("%w"), axis=1)
    data_appt_info['hour'] = data_appt_info.apply(lambda x: x.ScheduledStartTime.strftime("%H"), axis=1)

    print(f'\n Two type Duration')
    # 时长相关特征
    data_appt_info['Scheduled_duration'] = data_appt_info.apply(lambda x:
                                                                (
                                                                            x.ScheduledEndTime - x.ScheduledStartTime).seconds / 60,
                                                                axis=1)
    data_appt_info['Actual_duration'] = data_appt_info.apply(lambda x:
                                                             (x.ActualEndDate - x.ActualStartDate).seconds / 60, axis=1)

    data_appt_info.sort_values(by=['PatientSerNum', 'AppointmentSerNum'], inplace=True)

    return data_appt_info


def gettreat_info():
    logger.info('Get information about appointment')

    print('=' * 20)
    print(f'Get data from mysql')
    sql_radiation = """SELECT * FROM radiation"""
    data_radiation = get_df_from_sql(sql_radiation)
    print(f'The shape of data_radiation {data_radiation.shape}')
    sql_radiationhstry = """SELECT * FROM radiationhstry"""
    data_radiationhstry = get_df_from_sql(sql_radiationhstry)
    print(f'The shape of data_radiationhstry {data_radiationhstry.shape}')
    sql_plan = """SELECT * FROM plan"""
    data_plan = get_df_from_sql(sql_plan)
    print(f'The shape of data_plan {data_plan.shape}')
    sql_course = """SELECT * FROM course"""
    data_course = get_df_from_sql(sql_course)
    print(f'The shape of data_course {data_course.shape}')
    sql_patient = """SELECT * FROM patient"""
    data_patient = get_df_from_sql(sql_patient)
    print(f'The shape of data_patient {data_patient.shape}')

    try:
        print('Drop columns')
        data_radiation.drop('LastUpdated', axis=1, inplace=True)
        data_radiationhstry.drop('LastUpdated', axis=1, inplace=True)
        data_plan.drop('LastUpdated', axis=1, inplace=True)
        data_course.drop(columns=['LastUpdated'], inplace=True)
        data_patient.drop('LastUpdated', axis=1, inplace=True)
    except:
        print('Finish droppping columns')

    print('\nFinish getting data')

    print('=' * 20)
    print(f'Start merging data')

    # Radiation 的预处理
    print('\nProcess data_radiation')
    data_radiation = data_radiation[(data_radiation.DeliveryType == 'TREATMENT') &
                                    (data_radiation.MU > 0) &
                                    (data_radiation.MUCoeff > 0)]
    print(f'The shape of data_radiation {data_radiation.shape}')

    # Radiationhstry 的预处理
    print('\nProcess data_radiationhstry')
    data_radiationhstry = data_radiationhstry[
        data_radiationhstry.TreatmentStartTime > pd.Timestamp('2015-01-01 00:00:00')]
    print(f'The shape of data_radiationhstry {data_radiationhstry.shape}')

    print(f'\nMerge data_radiation and data_radiationhstry')
    ra_rh = pd.merge(data_radiation, data_radiationhstry,
                     on=['RadiationSerNum', 'AliasSerNum'], how='inner')
    print(f'The shape of ra_rh {ra_rh.shape}')
    del data_radiation, data_radiationhstry
    gc.collect

    print(f'\nMerge data_plan and ra_rh')
    ra_rh_pl = pd.merge(data_plan, ra_rh, on=['PlanSerNum', 'AliasSerNum'], how='inner')
    print(f'The shape of ra_rh_pl {ra_rh_pl.shape}')
    del data_plan, ra_rh
    gc.collect()

    print(f'\nMerge data_course and data_patient')
    co_pa = pd.merge(data_course, data_patient, on='PatientSerNum', how='inner')
    print(f'The shape of co_pa {co_pa.shape}')
    del data_course, data_patient
    gc.collect()

    print(f'\nMerge co_pa and ra_rh_pl')
    data_treat_info = pd.merge(co_pa, ra_rh_pl,
                               on=['CourseSerNum', 'AliasSerNum'], how='inner')
    print(f'The shape of data_treat_info {data_treat_info.shape}')
    del co_pa, ra_rh_pl
    gc.collect()

    # 　删除值完全相同的列
    print('=' * 20)
    print(f'Drop columns with same values')
    for col in data_treat_info.columns:
        if len(data_treat_info[col].unique()) == 1:
            data_treat_info.drop(col, axis=1, inplace=True)
    print(f'\ndata_treat_info shape {data_treat_info.shape}')

    print('\nFinish dropping')

    print('=' * 20)
    print(f'Create features')

    feature_columns = [
        'RadiationHstryAriaSer', 'TreatmentStartTime', 'TreatmentEndTime', 'FractionNumber', 'ImagesTaken', 'UserName',
        'RadiationSerNum', 'RadiationId', 'ResourceSerNum', 'MU', 'MUCoeff', 'TreatmentTime',
        'PatientSerNum',
        'CourseId'
    ]

    data_treat_info = data_treat_info[feature_columns]
    data_treat_info['date'] = data_treat_info.apply(lambda x: x.TreatmentStartTime.strftime("%Y--%m--%d"), axis=1)
    data_treat_info['Treatment_duration'] = data_treat_info.apply(lambda x:
                                                                  (x.TreatmentEndTime - x.TreatmentStartTime).seconds,
                                                                  axis=1)
    data_treat_info.sort_values(by=['PatientSerNum', 'RadiationHstryAriaSer'], inplace=True)

    return data_treat_info



if __name__ == '__main__':
    data = read_df_from_sql("""SELECT * FROM appointment""", WAITING_TIME_CONF)
