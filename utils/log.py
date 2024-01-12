
# -*- coding: utf-8 -*-
#  @file        - log.py
#  @author      - dongnian.wang(dongnian.wang@outlook.com)
#  @brief       - 日志模块， 实现print自动打印到日志
#  @version     - 0.0
#  @date        - 2023.12.20
#  @copyright   - Copyright (c) 2023 

""" 以下内容放在所有代码之前,实现print自动打印到日志
"""

import os,sys,time,io
import builtins as __builtin__

def print(*args, **kwargs):
    # __builtin__.print('New print function')
    return __builtin__.print("[", time.strftime("%Y-%m-%d %H:%M:%S ] ", time.localtime()) ,*args, **kwargs)

class Logger(object):
    def __init__(self, filename="default.log", path="./"):
        # sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        self.terminal = sys.stdout
        self.file_path = os.path.join(path, filename)
        self.log = open(self.file_path, "a", encoding='utf8')

    def write(self, message):
        self.terminal.write(message)
        self.log = open(self.file_path, "a", encoding='utf8')
        self.log.write(message)
        self.log.close()
        # self.log.flush()
        # self.terminal.flush()
        # sys.stdout.flush()

    def flush(self):
        pass


if __name__ == "__main__":
    # open log handler before print
    filename = "log"
    log_file_name = f"./{filename}.log"
    sys.stdout = Logger(str(log_file_name))

    a = 3
    print("1111111111", a)
    print("2222")
    print("444")