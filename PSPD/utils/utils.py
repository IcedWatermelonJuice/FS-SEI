import importlib
import os
import random
import sys
import time
import numpy as np
import torch
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
import datetime
import logging


def set_seed(seed):
    random.seed(seed)  # random
    np.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # GPU
    torch.cuda.manual_seed_all(seed)  # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False  # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现


def set_log_file_handler(log_dir):
    logging.getLogger().handlers = []
    logger = logging.getLogger(__name__)
    if logger.hasHandlers():
        logger.handlers.clear()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")

    strHandler = logging.StreamHandler()
    strHandler.setFormatter(formatter)
    logger.addHandler(strHandler)
    logger.setLevel(logging.INFO)

    log_file = os.path.join(log_dir, 'LOG_INFO.log')
    fileHandler = logging.FileHandler(log_file)
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    return logger


def get_logger_and_writer(root="./runs"):
    now_str = datetime.datetime.now().strftime('%m%d_%H%M%S')
    dir_path = os.path.join(root, now_str)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    logger = set_log_file_handler(dir_path)
    writer = SummaryWriter(os.path.join(dir_path, "writer"))
    return logger, writer, dir_path, root


def create_model(path, *args, **kwargs):
    module_name = "model_module"

    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module.create_model(*args, **kwargs)


class ListApply:
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        self.index = 0  # 初始化迭代索引
        return self

    def __next__(self):
        if self.index < len(self.data):
            result = self.data[self.index]
            self.index += 1
            return result
        else:
            raise StopIteration  # 迭代结束时抛出 StopIteration 异常

    def __getitem__(self, index):
        if 0 <= index < len(self.data):
            return self.data[index]
        else:
            raise IndexError("Index out of range")

    def __len__(self):
        return len(self.data)

    def __getattr__(self, method_name):
        # 检查列表中的每个元素是否具有相应的方法
        if all(hasattr(item, method_name) for item in self.data):
            # 定义一个新的方法，将其委托给列表中的每个元素执行
            def delegated_method(*args, **kwargs):
                return [getattr(item, method_name)(*args, **kwargs) for item in self.data]

            return delegated_method
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{method_name}'")


class RecordTime:
    def __init__(self, max_num=1):
        self.start_time = 0
        self.durations = []
        self.current_time = 0
        self.sum_time = 0
        self.max_num = max_num
        self.num = 0

    def start(self):
        self.start_time = time.time()

    def end(self):
        end_time = time.time()
        if self.start_time > 0:
            self.current_time = end_time - self.start_time
            self.durations.append(self.current_time)
            self.durations = self.durations[-10:]
            self.sum_time += self.current_time
            self.start_time = 0
            self.num += 1

    def mean_duration(self):
        return (self.sum_time / self.num) if self.num > 0 else 0

    def mean_last_10(self):
        return (sum(self.durations) / len(self.durations)) if len(self.durations) > 0 else 0

    def remaining_time(self):
        return (self.max_num - self.num) * self.mean_last_10()

    @staticmethod
    def format_time(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"[{hours:02d}:{minutes:02d}:{seconds:02d}]"

    def step(self):
        self.end()
        current_time = self.format_time(self.current_time)
        sum_time = self.format_time(self.sum_time)
        mean_time = self.format_time(self.mean_duration())
        remain_time = self.format_time(self.remaining_time())
        return current_time, mean_time, sum_time, remain_time


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = len(target)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res[0].item()

