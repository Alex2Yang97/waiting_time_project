"""
@Time: 2020/7/17 14:53
@Author: Zhirui(Alex) Yang
@E-mail: 1076830028@qq.com
@Program: database configuration
"""

import pymysql

# 数据库参数
WAITING_TIME_CONF = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'passwd': '980719',
    'db': 'waiting_time_data',
    'charset': 'utf8'
}


if __name__ == '__main__':
    con = pymysql.connect(
        host='localhost',
        port=3306,
        user='root',
        passwd='980719',
        db='waiting_time_data',
        charset='utf8')
    print('Connect successfully!')
    con.close()