#-*- coding:utf-8 -*-
# @Time     : 2020-07-11 18:13
# @Author   : Zhirui(Alex) Yang
# @Function :


import os
import logging
from datetime import datetime
from utils.config import LOG_LEVEL, DATA_DIR

logger = logging.getLogger("WAITING TIME")
level = logging.getLevelName(LOG_LEVEL)
logger.setLevel(level)
fmt = "WAITING TIME: %(asctime)s [%(levelname)s] %(message)s"
date_fmt = "%Y-%m-%d %H:%M:%S"
logger_path = os.path.join(DATA_DIR, 'log')


logging.basicConfig(format=fmt, datefmt=date_fmt)


if __name__ == '__main__':
    logger.info("Log configured successfully!")