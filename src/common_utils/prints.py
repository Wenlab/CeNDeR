# -*- coding: utf-8 -*-
#
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

import time

text_colors = {
    'logs'     : '\033[34m',  # 033 is the escape code and 34 is the color code
    'info'     : '\033[32m',
    'warning'  : '\033[33m',
    'error'    : '\033[31m',
    'bold'     : '\033[1m',
    'end_color': '\033[0m'
}


def timer(func):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print(f"{func.__name__} costs {t2 - t1} s")
        return result

    return wrapper


def get_curr_timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def get_checkpoint_timestamp():
    return time.strftime("%Y_%m_%d_%H_%M_%S")


def print_error_message(message):
    time_stamp = get_curr_timestamp()
    error_str = text_colors['error'] + text_colors['bold'] + 'ERROR  ' + text_colors['end_color']
    print('{} - {} - {}'.format(time_stamp, error_str, message))
    print('{} - {} - {}'.format(time_stamp, error_str, 'Exiting!!!'))
    exit(-1)


def print_log_message(message):
    time_stamp = get_curr_timestamp()
    log_str = text_colors['logs'] + text_colors['bold'] + 'LOGS   ' + text_colors['end_color']
    print('{} - {} - {}'.format(time_stamp, log_str, message))


def print_warning_message(message):
    time_stamp = get_curr_timestamp()
    warn_str = text_colors['warning'] + text_colors['bold'] + 'WARNING' + text_colors['end_color']
    print('{} - {} - {}'.format(time_stamp, warn_str, message))


def print_info_message(message):
    time_stamp = get_curr_timestamp()
    info_str = text_colors['info'] + text_colors['bold'] + 'INFO   ' + text_colors['end_color']
    print('{} - {} - {}'.format(time_stamp, info_str, message))


def pad_num(s, l = 3):
    s = str(s)
    _len = len(s)
    _l = l - _len
    return '0' * _l + s
