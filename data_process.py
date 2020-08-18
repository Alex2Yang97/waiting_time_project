#-*- coding:utf-8 -*-
# @Time     : 2020-07-11 18:44
# @Author   : Zhirui(Alex) Yang
# @Function :

import gc
import os
import datetime

import pandas as pd
from utils.logger import logger
from utils.funcs import read_df_from_sql, drop_cols_with_same_values


APPOINTMENT_FEATURE = [
    'dxt_AliasName',
    'DateOfBirth', 'Sex',
    'AliasSerNum', 'PatientSerNum', 'AppointmentSerNum',
    'ScheduledStartTime', 'ScheduledEndTime', 'ActualStartDate', 'ActualEndDate',
    'CourseSerNum',
    'DoctorSerNum',
    'PlanSerNum',
    'TreatmentOrientation',
    ]


TREATMENT_FEATURE = [
    'RadiationHstryAriaSer',
    'TreatmentStartTime', 'TreatmentEndTime', 'ImagesTaken', 'TreatmentTime',
    'FractionNumber', 'UserName',
    'RadiationSerNum', 'ResourceSerNum', 'CourseId',
    'MU', 'MUCoeff',
    'PatientSerNum'
    ] # RadiationId


def get_apptointment_info():
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
    merged_data = pd.merge(data_appointment, data_pt, on='PatientSerNum', how='inner')
    merged_data = pd.merge(merged_data, data_ptc, on='PatientSerNum', how='inner')
    del data_appointment, data_pt, data_ptc
    gc.collect()

    logger.debug('Get table of patientdoctor!')
    data_pd = read_df_from_sql('SELECT * FROM patientdoctor').drop(columns=['LastUpdated'])
    data_pd = data_pd[(data_pd.OncologistFlag == 1) & (data_pd.PrimaryFlag == 1)]

    logger.debug('Merge patientdoctor!')
    # data_pd, data_appt AliasSerNum 不同，data_appt_ 中只保留了23、31，data_pd_ 中只有37
    data_pd = data_pd.drop(columns=['AliasSerNum'])
    merged_data = pd.merge(merged_data, data_pd, on='PatientSerNum', how='left')
    del data_pd
    gc.collect()

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
    merged_data = pd.merge(merged_data, dx_dxt, on=['DiagnosisSerNum'], how='inner')

    logger.debug('Get table of course!')
    data_co = read_df_from_sql('SELECT * FROM course')
    data_co = data_co.drop(columns=['AliasSerNum'])

    logger.debug('Get table of plan!')
    data_pl = read_df_from_sql('SELECT * FROM plan')
    data_pl = data_pl[data_pl.Status == 'TreatApproval']
    data_pl['TreatmentOrientation'] = data_pl['TreatmentOrientation'].apply(
        lambda x: 'NULL' if x == '' else x)
    data_pl = data_pl.drop(columns=['AliasSerNum'])

    logger.debug('Merge plan and course!')
    pl_co = pd.merge(data_pl, data_co, on='CourseSerNum', how='inner')
    pl_co = pl_co.drop(columns=['AliasExpressionSerNum', 'DiagnosisSerNum', 'PrioritySerNum'])
    pl_co = pl_co.rename(columns={'Status': 'Status_plan'})
    del data_pl, data_co
    gc.collect()

    logger.debug('Merge plan_course and appointment!')
    merged_data = pd.merge(merged_data, pl_co, on=['PatientSerNum'], how='inner')
    del pl_co
    gc.collect()

    logger.debug(f'Drop columns with same values!')
    merged_data = drop_cols_with_same_values(merged_data)

    logger.debug('Process appointment data!')
    processed_appt_data = merged_data[APPOINTMENT_FEATURE]

    processed_appt_data['age'] = processed_appt_data.apply(
        lambda x: int((x.ActualStartDate - x.DateOfBirth).days / 365), axis=1)
    processed_appt_data['month'] = processed_appt_data.apply(lambda x: x.ScheduledStartTime.strftime("%m"), axis=1)
    processed_appt_data['date'] = processed_appt_data.apply(lambda x: x.ScheduledStartTime.strftime("%Y--%m--%d"), axis=1)
    processed_appt_data['week'] = processed_appt_data.apply(lambda x: x.ScheduledStartTime.strftime("%w"), axis=1)
    processed_appt_data['hour'] = processed_appt_data.apply(lambda x: x.ScheduledStartTime.strftime("%H"), axis=1)
    processed_appt_data['Scheduled_duration'] = processed_appt_data.apply(
        lambda x: (x.ScheduledEndTime - x.ScheduledStartTime).seconds / 60, axis=1)
    processed_appt_data['Actual_duration'] = processed_appt_data.apply(
        lambda x: (x.ActualEndDate - x.ActualStartDate).seconds / 60, axis=1)
    processed_appt_data = processed_appt_data.sort_values(by=['PatientSerNum', 'AppointmentSerNum'])
    processed_appt_data = processed_appt_data.drop_duplicates().reset_index(drop=True)

    return processed_appt_data


def get_treat_info():
    logger.info('Get information about treatment!')

    logger.debug('Get table of radiation!')
    data_radiation = read_df_from_sql('SELECT * FROM radiation').drop(columns=['LastUpdated'])
    data_radiation = data_radiation[
        (data_radiation.DeliveryType == 'TREATMENT')
        & (data_radiation.MU > 0)
        & (data_radiation.MUCoeff > 0)]

    logger.debug('Get table of radiationhstry!')
    data_radiationhstry = read_df_from_sql('SELECT * FROM radiationhstry').drop(columns=['LastUpdated'])
    data_radiationhstry = data_radiationhstry[
        data_radiationhstry.TreatmentStartTime > pd.Timestamp('2015-01-01 00:00:00')]

    logger.debug('Merge radiation and radiationhstry!')
    merged_data = pd.merge(data_radiation, data_radiationhstry,
                           on=['RadiationSerNum', 'AliasSerNum'], how='inner')
    del data_radiation, data_radiationhstry
    gc.collect()

    logger.debug('Merge plan!')
    data_plan = read_df_from_sql('SELECT * FROM plan').drop(columns=['LastUpdated'])
    merged_data = pd.merge(data_plan, merged_data, on=['PlanSerNum', 'AliasSerNum'], how='inner')
    del data_plan
    gc.collect()

    logger.debug('Get table of radiation!')
    data_course = read_df_from_sql('SELECT * FROM course').drop(columns=['LastUpdated'])

    logger.debug('Get table of radiation!')
    data_patient = read_df_from_sql('SELECT * FROM patient').drop(columns=['LastUpdated'])

    logger.debug('Merge course and patient')
    co_pa = pd.merge(data_course, data_patient, on='PatientSerNum', how='inner')
    del data_course, data_patient
    gc.collect()

    logger.debug('Merge radiation and radiation!')
    merged_data = pd.merge(co_pa, merged_data,
                           on=['CourseSerNum', 'AliasSerNum'], how='inner')
    del co_pa
    gc.collect()

    logger.debug(f'Drop columns with same values!')
    merged_data = drop_cols_with_same_values(merged_data)

    logger.debug('Process treatment data!')
    processed_treat_data = merged_data[TREATMENT_FEATURE]
    processed_treat_data['date'] = processed_treat_data.apply(
        lambda x: x.TreatmentStartTime.strftime("%Y--%m--%d"), axis=1)
    processed_treat_data['Treatment_duration'] = processed_treat_data.apply(
        lambda x: (x.TreatmentEndTime - x.TreatmentStartTime).seconds, axis=1)
    processed_treat_data = processed_treat_data.sort_values(by=['PatientSerNum', 'RadiationHstryAriaSer'])
    processed_treat_data = processed_treat_data.drop_duplicates().reset_index(drop=True)

    return processed_treat_data


# 因为部分特征有多个取值，因此对于这样的数据，如果在DataFrame 中变成一个list
# 一个取值的，还是值，而不是list
def get_list(x):
    x = list(x)
    if len(x) == 1:
        return x[0]
    else:
        return x


if __name__ == '__main__':
    processed_appointment_data = get_apptointment_info()
    print(f'process_appointment_data shap {processed_appointment_data.shape}')
    processed_treatment_data = get_treat_info()
    print(f'process_treatment_data shap {processed_treatment_data.shape}')