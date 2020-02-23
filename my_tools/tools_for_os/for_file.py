#-*- coding:utf-8 -*-
# @Time     : 2020-02-15 15:49
# @Author   : Zhirui(Alex) Yang
# @Function :


import os


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(folder_name, 'has created!')
    else:
        print(folder_name, 'already existed!')


