#-*- coding:utf-8 -*-
# @Time     : 2020-07-11 18:15
# @Author   : Zhirui(Alex) Yang
# @Function :


import os

# 日志等级
LOG_LEVEL = 'DEBUG'

# 路径
__proj_dir = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(__proj_dir, 'data')


MODEL_DIR = os.path.join(DATA_DIR, 'model')
RES_DIR = os.path.join(DATA_DIR, 'result')
FIG_DIR = os.path.join(DATA_DIR, 'figure')


if __name__ == '__main__':
    print(__proj_dir)
    print(DATA_DIR)