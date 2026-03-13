import os
import json
import numpy as np
from copy import copy, deepcopy
from matplotlib import pyplot as plt
from typing import List, Dict, Any, AnyStr, Literal, Iterator, Callable
from contextlib import contextmanager

from utils.COLOR import printc
from utils.GetPath import *

TimeXer_Path = "TimeXer_loss.json"
CEEFD_Path = "CEEFD_TimeXer_loss.json"

TimeXer_Random_Path = "TimeXer_loss_RandomSplit.json"
CEEFD_Random_Path = "CEEFD_TimeXer_loss_RandomSplit.json"

@contextmanager
def control_show(is_show: bool = True, is_save: bool = False) -> None:
    origin_show = plt.show
    origin_savefig = plt.savefig

    if not is_show:
        plt.show = lambda *args, **kwargs: None   # 抑制 plt.show，这样写函数时可以直接写 plt.plt.show 不需要在函数形参中接收 is_show 参数进行内部的修改用于控制 plt.show 的行为

    if not is_save:
        plt.savefig = lambda *args, **kwargs: None

    try:
        yield

    finally:
        if plt.show is None:
            plt.show = origin_show

        if plt.savefig is None:
            plt.savefig = origin_savefig

    return


def get_loss_data(path) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        file = json.load(f)

    return file

def pict(*path) -> bool:
    try:
        if len(path) == 1:
            path = path[0]

            file = get_loss_data(path)

            Train_Loss = file["train_loss"]
            Val_Loss = file["val_loss"]

            x = range(1, len(Train_Loss) + 1)

            # Train
            plt.plot(x, Train_Loss, color="blue", linestyle="--", label="Train")
            plt.plot(x, Val_Loss, color="red", linestyle="-", label="Val")

            plt.legend(loc=1)
            plt.xlim((1, 100))
            plt.tight_layout()

            path: str
            _pth = path.split('_')[0]

            _tpth = os.path.join("Pictures", f"{_pth}_loss.png")
            plt.savefig(_tpth, dpi=300)
            plt.show()

        elif len(path) >= 2:
            colors = [["b", "r"], ["g", "b"]]  # 第一组 | 第二组 -> [Train | Val]
            linestyles = ["--", "-"]
            labels = ["Train", "Val"]

            for idx_pth, pth in enumerate(path):
                _data: dict = get_loss_data(pth)
                # print(_data.items())

                for idx, d in enumerate(_data.values()):
                    x = range(1, len(d) + 1)

                    if idx_pth <= 1:
                        _color = colors[idx_pth][idx]
                    else:
                        _color = 'r'

                    # plt.plot(x, d)
                    plt.plot(x, d, color=_color, linestyle=linestyles[idx], label=labels[idx])

                plt.legend(loc=1)
                plt.xlim((1, 100))
                plt.tight_layout()

                pth: str
                _pth = pth.split('_')[0]
                name = _pth.split('_')[0]

                plt.title(name)

                _tpth = os.path.join("Pictures", f"{_pth}_loss.png")
                plt.savefig(_tpth, dpi=300)
                plt.show()

            p1 = None  # Pure
            p2 = None  # CEEFD
            for p in path:
                if "ceefd" in str(p).lower():
                    p2 = p

                else:
                    p1 = p

            # print(path, p1, p2)
            Pure_df = get_json_data(p1)
            CEEFD_df = get_json_data(p2)

            Train_loss = [Pure_df["train_loss"], CEEFD_df["train_loss"]]
            Val_loss = [Pure_df["val_loss"], CEEFD_df["val_loss"]]

            x = range(1, 101)

            plt.subplot(1, 2, 1)
            plt.plot(x, Train_loss[0], c='r', label="Pure-Train")
            plt.plot(x, Train_loss[1], c='blue', label="CEEFD-Train")
            plt.legend(loc=1)

            plt.subplot(1, 2, 2)
            plt.plot(x, Val_loss[0], c='r', label="Pure-Val")
            plt.plot(x, Val_loss[1], c='blue', label="CEEFD-Val")
            plt.legend(loc=1)

            _tpth = os.path.join("Pictures", "Comparison.png")
            plt.savefig(_tpth, dpi=300)

            plt.show()

    except Exception as e:
        printc(f"报错：{e}", color="magenta")
        raise

        return False


if __name__ == '__main__':
    # Pure_df = get_json_data(TimeXer_Random_Path)
    # CEEFD_df = get_json_data(CEEFD_Random_Path)
    #
    # Train_loss = [Pure_df["train_loss"], CEEFD_df["train_loss"]]
    # Val_loss = [Pure_df["val_loss"], CEEFD_df["val_loss"]]
    #
    # x = range(1, 101)
    #
    # plt.subplot(1, 2, 1)
    # plt.plot(x, Train_loss[0], c='r', label="Pure-Train")
    # plt.plot(x, Train_loss[1], c='blue', label="CEEFD-Train")
    # plt.legend(loc=1)
    #
    # plt.subplot(1, 2, 2)
    # plt.plot(x, Val_loss[0], c='r', label="Pure-Val")
    # plt.plot(x, Val_loss[1], c='blue', label="CEEFD-Val")
    # plt.legend(loc=1)
    #
    # plt.show()

    # with control_show(True, True):
    #     # pict("TimeXer_loss_RandomSplit.json", "CEEFD_TimeXer_loss_RandomSplit.json")
    #     # pict("CEEFD_TimeXer_loss_RandomSplit.json")
    #     # pict(CEEFD_Path)
    #     pict(TimeXer_Random_Path, CEEFD_Random_Path)

    lower = get_json_data("TimeXer_loss_d_ff_256.json")
    better = get_json_data("TimeXer_loss_RandomSplit.json")
    ceefd = get_json_data("CEEFD_TimeXer_loss_RandomSplit.json")

    datas = (lower, better, ceefd)

    Train = []
    Val = []

    for idx, d in enumerate(datas):
        if idx >= 1:
            _tmp_train = []
            _tmp_val = []

            _tmp_train.extend(d["train_loss"])
            _tmp_val.extend(d["val_loss"])

            Train.append(_tmp_train)
            Val.append(_tmp_val)

        else:
            Train.append(d["train_loss"])
            Val.append(d["val_loss"])

    x = range(1, len(Train[0]) + 1)
    _x = range(100, 200)

    labels = ["d_ff=256", "d_ff=1024", "ceefd"]
    colors = ["r","b","g"]

    plt.figure()
    for i in range(3):
        plt.plot(x if i == 0 else _x, Train[i], label=labels[i], color=colors[i])

    plt.legend(loc=1)
    plt.title("Train")
    plt.show()

    plt.figure()
    for i in range(3):
        plt.plot(x if i == 0 else _x, Val[i], label=labels[i], color=colors[i])

    plt.legend(loc=1)
    plt.title("Val")
    plt.show()