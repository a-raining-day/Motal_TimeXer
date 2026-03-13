import pickle
from typing import Any

import pandas as pd
import polars as pl
import numpy as np
from matplotlib import pyplot as plt
from numpy import signedinteger

from utils.COLOR import printc

from 模态分解.CEEFD import *
from 模态分解.EFD import *

from give_data import *

ceefd = CEEFD()

modals = [ceefd.ceefd, ceefd.ceemdan, EFD]
names = ["ceefd", "ceemdan", "efd"]

def energy(seq: np.ndarray) -> signedinteger:
    # print(seq.shape)

    if seq.shape[0] == 1 or np.ndim(seq) == 1:
        return np.sum(seq ** 2)

    else:
        # print(seq)
        return np.sum(seq ** 2, axis=1)

def main(modal, name):
    printc(f"==={name}===", color="red")
    Metrix, Total_Time, names, each_column_start_and_end = Metrix_Create()
    Metrix: np.ndarray

    en_seq = Metrix[1, :][~np.isnan(Metrix[1, :])]

    if name == "ceefd":
        other_IMFs, IMF_, Res, Res_ = modal(en_seq)
        IMF_: List[np.ndarray]
        IMF_ = sorted(IMF_, key=lambda x: energy(x), reverse=True)
        # print(IMF_)
        # raise
        IMF_: np.ndarray = np.stack(IMF_)
        Res = Res.reshape(1, -1)
        Res_ = Res_.reshape(1, -1)

        # print(IMF_)
        # raise

        print(other_IMFs.shape[0], IMF_.shape[0], Res.shape[0], Res_.shape[0])
        # print(IMF_)
        # raise

        print(f"energy | other_IMFs: {energy(other_IMFs)}, IMF_: {energy(IMF_)}, Res: {energy(Res)}, Res_: {energy(Res_)}")
        print(f"other_IMFs: {energy(other_IMFs).shape}, IMF_: {energy(IMF_).shape}, Res: {energy(Res).shape}, Res_: {energy(Res_).shape}")

        d = {"IMF": IMF_.tolist(), "other_IMFs": other_IMFs.tolist(), "Res": Res.tolist(), "Res_": Res_.tolist()}

        with open(f"{name}.pkl", "wb") as f:
            pickle.dump(d, f)

    else:
        IMFs, Res = modal(en_seq)
        IMFs = np.array(IMFs)
        Res = np.array(Res).reshape(1, -1)

        print(IMFs.shape[0], Res.shape[0])

        print(f"energy | IMFs: {energy(IMFs)}, Res: {energy(Res)}")
        print(f"IMFs: {energy(IMFs).shape}, Res: {energy(Res).shape}")

        d = {"IMFs": IMFs, "Res": Res.tolist()}

        with open(f"{name}.pkl", 'wb') as f:
            pickle.dump(d, f)

    print('\n')

def plot(name, labelpad=15, fontsize=10):
    printc(f"==={name}===", color="red")
    Metrix, Total_Time, _names, each_column_start_and_end = Metrix_Create()
    Metrix: np.ndarray

    en_seq = Metrix[1, :][~np.isnan(Metrix[1, :])]
    N = len(en_seq)
    T = np.arange(N)

    with open(f"{name}.pkl", 'rb') as f:
        file = pickle.load(f)

    if name == "ceefd":
        other_IMFs = np.stack(file["other_IMFs"])
        IMF_ = np.stack(file["IMF"])[:3, :]
        Res = np.array(file["Res"]).reshape(1, -1)
        # Res_ = np.array(file["Res_"]).reshape(1, -1)

        # print(other_IMFs.shape, IMF_.shape)
        # raise
        col = 1 + other_IMFs.shape[0] + IMF_.shape[0] + 1  # origin | other_imfs | imf | res

        # print(other_IMFs.shape, IMF_.shape, Res.shape)

        total = np.vstack((en_seq, other_IMFs, IMF_, Res))
        # print(total.shape)
        # raise

        plt.subplots(col, 1, figsize=(15, 50))
        plt.suptitle("CEEFD", y=0.89)
        for i in range(col):
            c = "red" if i == 0 else "#20B2AA"
            # print(total.shape[0])
            # raise
            plt.subplot(total.shape[0], 1, i + 1)
            plt.plot(T, total[i, :], color=c, linewidth=1)
            plt.xticks([])
            if i != col - 1 and i != 0:
                plt.ylabel(f"IMF{i}", rotation=0, labelpad=labelpad, fontsize=fontsize)

            elif i == 0:
                plt.ylabel("origin", rotation=0, labelpad=labelpad, fontsize=fontsize)

            else:
                plt.ylabel(f"Residual", rotation=0, labelpad=labelpad, fontsize=fontsize)

        plt.savefig(f"{name}.svg", dpi=300)

        # plt.tight_layout()
        plt.show()

    elif name == "ceemdan":
        IMFs = file["IMFs"]
        Res = file["Res"]

        IMFs = np.stack(IMFs)
        Res = np.array(Res)

        total = np.vstack((en_seq, IMFs, Res))

        col = total.shape[0]

        plt.subplots(col, 1, figsize=(15, 30))
        plt.suptitle("CEEMDAN", y=0.89)
        for i in range(col):
            c = "red" if i == 0 else "#20B2AA"

            plt.subplot(col, 1, i + 1)
            plt.plot(T, total[i, :], linewidth=1, color=c)

            plt.xticks([])
            if i == 0:
                plt.ylabel("origin", labelpad=labelpad, rotation=0, fontsize=fontsize)

            elif 0 < i < col - 1:
                plt.ylabel(f"IMF{i}", labelpad=labelpad, rotation=0, fontsize=fontsize)

            else:
                plt.ylabel("Residual", labelpad=labelpad, rotation=0, fontsize=fontsize)

        plt.savefig(f"{name}.svg", dpi=300)
        plt.show()

    else:
        IMFs = file["IMFs"]
        Res = file["Res"]

        IMFs = np.stack(IMFs)[:3, :]
        Res = np.array(Res)

        total = np.vstack((en_seq, IMFs, Res))

        col = total.shape[0]

        plt.subplots(col, 1, figsize=(15, 30))
        plt.suptitle("EFD", y=0.89)
        for i in range(col):
            c = "red" if i == 0 else "#20B2AA"

            plt.subplot(col, 1, i + 1)
            plt.plot(T, total[i, :], linewidth=1, color=c)

            plt.xticks([])
            if i == 0:
                plt.ylabel("origin", rotation=0, labelpad=labelpad, fontsize=fontsize)

            elif 0 < i < col - 1:
                plt.ylabel(f"IMF{i}", rotation=0, labelpad=labelpad, fontsize=fontsize)

            else:
                plt.ylabel("Residual", rotation=0, labelpad=10, fontsize=fontsize)

        plt.savefig(f"{name}.svg", dpi=300)
        plt.show()

    print('\n')

if __name__ == '__main__':
    # for i in range(3):
    #     main(modals[i], names[i])

    for i in range(3):
        plot(names[i])

    # plot(names[2])
