import os
import polars as pl
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path, WindowsPath
from typing import AnyStr, List, Union, Literal, Callable

__all__ = ["get_time"]

def get_time(path_ls: List[Union[AnyStr, WindowsPath, Path]], formation: Literal['/', '-'] = '/', cover: bool=False) -> List[datetime]:
    """
    获取统一的时间轴
    """

    if not isinstance(path_ls, list):
        raise TypeError("path_ls 必须为 list!")

    # 先避免一下有两个括号的情况
    if len(path_ls) == 1:
        if isinstance(path_ls, list):
            raise ValueError("格式错误！path_ls 应该是一个一维列表！")

    def time_f(formation: Literal['/', '-']) -> Callable[[str], str]:
        if formation == '/':
            def _f(date: str) -> str:
                if '-' in date:
                    return '/'.join(date.split('-'))

        else:
            def _f(date: str) -> str:
                if '/' in date:
                    return '-'.join(date.split('/'))

        return _f

    Time = []
    for pth in path_ls:
        file = pl.read_csv(pth, infer_schema_length=10000)

        file = file.with_columns \
            (
                pl.col(file.columns[0]).map_elements(time_f(formation), return_dtype=str)
            )

        Time.extend(file[:, 0].to_numpy().tolist())

    Time = sorted(list(set(Time)))
    def trans(date: str):
        if '/' in date:
            year, month, day = date.split('/')

        if '-' in date:
            year, month, day = date.split('-')

        return pd.Timestamp(year=int(year), month=int(month), day=int(day)).to_pydatetime()

    Time = sorted([trans(date) for date in Time])

    return Time

if __name__ == '__main__':
    from utils.GetPath import get_path_list

    ls = get_path_list("Data")
    get_time(ls)