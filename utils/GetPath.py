import os
import json
import pathlib
import polars as pl
from typing import Union, Dict, List, AnyStr, Iterator, Tuple, Generator
from .CONST_PATH import *
from .EEROR import DontKnownError
from warnings import warn

def isNone(var) -> bool:
    if var is None:
        return True

    else:
        return False

def notNone(var) -> bool:
    if var is not None:
        return True

    else:
        return False

class EXIST:
    # 两类：路径检测 和 变量定义检测

    @staticmethod
    def is_exist_GlobalVar(var_name: str) -> bool:
        """在当前全局变量中检查"""

        if var_name in globals():
            return True

        else:
            return False

    @staticmethod
    def is_exist_ConstVar(var_name: str) -> bool:
        """检测 CONST_PATH 的变量"""
        if var_name in Global:
            return True

        else:
            return False

    @staticmethod
    def continue_exist_ConstVar(var_name: str) -> bool:
        """从 CONST_PATH 中一直检测到 var_name 为止"""
        l = len(Base.pathList)
        # l = len(GlobalKeys)

        if l != len(GlobalKeys):
            return False

        for i in range(l):
            if Base.pathList[i] != GlobalKeys[i]:
                return False

        return True

    @staticmethod
    def is_exist_Data_dir() -> bool:
        if os.path.exists(PATH.get_Data_path()):
            return True

        else:
            return False


class PATH:
    # 获取路径

    @staticmethod
    def get_data_path() -> AnyStr:
        if EXIST.is_exist_ConstVar("path_data"):
            return path_data

        else:
            data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
            return data_path

    @staticmethod
    def get_Data_path() -> Union[None, AnyStr]:
        """得到 Data 文件夹路径"""

        data_path = PATH.get_data_path()

        if not isinstance(data_path, str):
            return None

        Data_path = os.path.join(data_path, "Data")

        return Data_path

def a_and_b(a: bool, b: bool) -> bool:
    if a is None:
        a = False

    if b is None:
        b = False

    if a and b:
        return True

    else:
        return False

def  a_and_unb(a: bool, b: bool) -> bool:
    if a is None:
        a = False

    if b is None:
        b = False

    if a and not b:
        return True

    else:
        return False

def una_and_b(a: bool, b: bool) -> bool:
    if a is None:
        a = False

    if b is None:
        b = False

    if not a and b:
        return True

    else:
        return False

def una_and_unb(a: bool, b: bool) -> bool:
    if a is None:
        a = False

    if b is None:
        b = False

    if not a and not b:
        return True

    else:
        return False

def get_path_iter(path) -> Iterator:
    return pathlib.Path(path).iterdir()

def get_json_data(json_path) -> dict:
    """
    返回一个来自 json 的字典
    :param json_path: json 文件路径，只能是 utf-8 编码
    :return:
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        df = json.load(f)

    return df

def get_path_list \
    (
        path: Union[str, pathlib.Path],
        is_all: bool = None,
        files_only: bool = None
    ) -> List[pathlib.Path]:

    """
    获取指定路径下的条目列表。

    Args:
        path: 起始目录路径
        is_all: 是否递归遍历子目录
        files_only: 是否只返回文件（忽略目录下的子文件夹）

    Returns:
        包含 pathlib.Path 对象的列表
    """

    if is_all == files_only:
        if is_all == files_only is None:
            # 默认为返回下属的所有文件和文件夹
            is_all = True
            files_only = False

        else:
            if is_all == files_only == True:
                raise ValueError("is_all 参数和 files_only 参数不能同时为 True！")

    if is_all is None:
        is_all = False

    if files_only is None:
        files_only = False

    root = pathlib.Path(path)
    if not root.is_dir():
        raise TypeError(f"应当为文件夹目录，实际为 {path}")

    result = []
    stack = [root]  # 用栈模拟深度优先遍历

    if files_only:
        current = stack.pop()
        try:
            # 获取当前目录下的所有条目
            entries = list(current.iterdir())
        except PermissionError:
            return []  # 忽略无权限访问的目录

        for entry in entries:
            if entry.is_file():
                result.append(entry)

        return result

    elif not files_only and not is_all:
        current = stack.pop()
        try:
            # 获取当前目录下的所有条目
            entries = list(current.iterdir())
        except PermissionError:
            return []  # 忽略无权限访问的目录

        for entry in entries:
            result.append(entry)

        return result

    while stack:  # 这里一定是 is_all == True
        current = stack.pop()
        try:
            # 获取当前目录下的所有条目
            entries = list(current.iterdir())
        except PermissionError:
            continue  # 忽略无权限访问的目录

        for entry in entries:
            is_file = entry.is_file()
            is_dir = entry.is_dir()

            if is_file:
                result.append(entry)

            # 如果需要递归且是目录，加入栈
            if is_all and is_dir:
                stack.append(entry)

    return result

def get_Data(isIterator: bool = True, isPath: bool = False) -> \
    Union[None, Generator[pathlib.Path, None, None], pathlib.Path, tuple[pathlib.Path, Generator[pathlib.Path, None, None]]]:
    """返回一个迭代器 或者 返回一个 pathlib.Path 对象"""

    _path = PATH.get_Data_path()
    if _path is None:
        return None

    _path_iter = pathlib.Path(_path)

    if isIterator and not isPath:
        return _path_iter.iterdir()

    if not isIterator and isPath:
        return _path_iter

    if isIterator and isPath:
        return _path_iter, _path_iter.iterdir()

    if not isIterator and not isPath:
        return None

def get_Data_path_list(isIterator: bool=True, isPath: bool=False, ls_is_str: bool = False) -> Union[None, list]:
    """
    尽量不要用这个，会加载处理所有的一级文件，类型只有文件\n
    现在改了，直接返回一个列表，这些选择参数本来就不必要\n
    返回一个 Data 文件夹下的各个文件的列表
    :param isIterator: 维持原接口
    :param isPath: 维持原接口
    :return:
    """
    Data_path = PATH.get_Data_path()

    if Data_path is None or una_and_unb(isIterator, isPath):
        warn("Data_path is None!")
        return None

    files = get_path_list(Data_path, True, False)
    if ls_is_str:
        files = [str(pth) for pth in files]
    return files

# print(get_Data_path_list())

def get_Data_num(isIterator: bool=True, isPath: bool=False) -> Union[None, int]:
    files = get_Data_path_list(isIterator, isPath)

    if files is None:
        return None

    return len(files)