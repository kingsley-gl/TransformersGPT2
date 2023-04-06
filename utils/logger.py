#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2023/3/9 17:17
# @Author  : kingsley kwong
# @Site    :
# @File    : logger.py
# @Software: 
# @Function:

import logging
import logging.config
import os

BASE_DIR = os.path.dirname(os.path.abspath(__package__))
TRAIN_DIR = '/'.join([BASE_DIR, '/train/log'])
PREPROCESS_DIR = '/'.join([BASE_DIR, '/preprocess/log'])

LOGGERS = {
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(threadName)s:%(thread)d] [%(name)s:%(lineno)d] [%(module)s:%(funcName)s] [%(levelname)s]- %(message)s'}
        # 日志格式
    },
    'filters': {
    },
    'handlers': {
        'trainError': {
            'level': 'ERROR',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '//'.join([TRAIN_DIR, 'error.log']),
            'maxBytes': 1024 * 1024 * 5,
            'backupCount': 5,
            'formatter': 'standard',
        },
        'trainInfo': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',  # 按文件大小
            'filename': '//'.join([TRAIN_DIR, 'info.log']),
            'maxBytes': 1024 * 1024 * 5,
            'backupCount': 5,
            'formatter': 'standard',
        },
        'preprocessError': {
            'level': 'ERROR',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '//'.join([PREPROCESS_DIR, 'error.log']),
            'maxBytes': 1024 * 1024 * 5,
            'backupCount': 5,
            'formatter': 'standard',
        },
        'preprocessInfo': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',  # 按文件大小
            'filename': '//'.join([PREPROCESS_DIR, 'info.log']),
            'maxBytes': 1024 * 1024 * 5,
            'backupCount': 5,
            'formatter': 'standard',
        },
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'standard'
        }
    },
    'loggers': {
        'train': {'handlers': ['console', 'trainError', 'trainInfo'],
                  'level': 'INFO',
                  'propagate': True},
        'preprocess': {'handlers': ['console', 'preprocessError', 'preprocessInfo'],
                       'level': 'INFO',
                       'propagate': True},
        'interact': {'handlers': ['console'],
                       'level': 'INFO',
                       'propagate': True}
    }
}


def get_logger(name):
    if not os.path.exists(TRAIN_DIR):
        os.makedirs(TRAIN_DIR)
    if not os.path.exists(PREPROCESS_DIR):
        os.makedirs(PREPROCESS_DIR)
    logging.config.dictConfig(LOGGERS)
    logger = logging.getLogger(name)
    return logger
