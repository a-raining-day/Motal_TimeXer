"""所有访问 CONST_PATH 的渠道只能从 GetPath 进入"""

import os

Global = globals()
GlobalKeys = Global.keys()
GlobalValue = Global.values()

class Base:
    pathList = \
    [
        "path_Current",
        "path_Parent_Current",
        "dir_Parent_Current",
        "path_data"
    ]

    varList = None

path_Current = __file__  # 当前文件的路径
path_Parent_Current = os.path.dirname(path_Current)  # 当前文件的文件夹
dir_Parent_Current = os.path.dirname(path_Parent_Current)  # 当前项目的目录
path_data = os.path.join(dir_Parent_Current, "data")
