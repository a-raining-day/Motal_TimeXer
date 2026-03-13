import os
import numpy as np
import polars as pl
from utils.GetPath import *

path_ls = get_Data_path_list()

columns = 0
length = []
for pth in path_ls:
    file = pl.read_csv(pth, infer_schema_length=10000)

    _shape = file.shape

    columns += _shape[1] - 1
    length.append(_shape[0] - 1)


length_dict = {}
for l in length:
    length_dict[l] = length_dict.get(l, 0) + 1

print(f"各数据文件的列数：{columns}")
print(f"各序列长度：{length}")
print(f"平均长度：{np.mean(length)}")

for k, v in length_dict.items():
    print(f"{k}  -  {v}")

def up_num(l: int) -> int:
    count = 0
    for i in length:
        if i >= l:
            count += 1

    print(f"大于 {l} 的序列有：{count} 条")

    return count

# up_num(3000)
up_num(4000)
# up_num(5000)