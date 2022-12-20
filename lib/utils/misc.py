import functools
from termcolor import colored

bar_perfixes = {
    "train": colored("train", "white", attrs=["bold"]),
    "val": colored("val", "yellow", attrs=["bold"]),
    "test": colored("test", "magenta", attrs=["bold"]),
}


def param_size(net):
    # ! treat all parameters to be float
    return sum(p.numel() for p in net.parameters()) * 4 / (1024 * 1024)


def singleton(cls):
    _instance = {}

    @functools.wraps(cls)
    def inner(*args, **kwargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]
    return inner
