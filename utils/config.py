#-*- coding:utf-8 -*-
# @Time     : 2020-07-11 18:15
# @Author   : Zhirui(Alex) Yang
# @Function :


import os

# 日志等级
LOG_LEVEL = 'DEBUG'

# 路径
__proj_dir = os.path.dirname(os.path.dirname(__file__))
__data_dir = os.path.join(__proj_dir, 'data')


MODEL_DIR = os.path.join(__data_dir, 'model')
RES_DIR = os.path.join(__data_dir, 'result')
FIG_DIR = os.path.join(__data_dir, 'figure')


if __name__ == '__main__':
    print(__proj_dir)
    print(__data_dir)