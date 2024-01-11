#以下内容放在所有代码之前,实现print自动打印到日志
import os,sys,time,io
import builtins as __builtin__
def print(*args, **kwargs):
    # __builtin__.print('New print function')
    return __builtin__.print("[", time.strftime("%Y-%m-%d %H:%M:%S ] ", time.localtime()) ,*args, **kwargs)
class Logger(object):
    def __init__(self, filename="Default.log", path="./"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        self.terminal = sys.stdout
        self.log = open(os.path.join(path, filename), "a", encoding='utf8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


# # open log handler before print
# filename = "log"
# log_file_name = f"long/{filename}.log"
# sys.stdout = Logger(str(log_file_name))