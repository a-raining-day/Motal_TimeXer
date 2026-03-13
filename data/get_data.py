import os
import json
import polars as pl
from typing import List, Dict, Union, Any

from polars import DataFrame


def get_data \
    (
        # filter_weekend: bool=False,
        only_sequence: bool=True
    ) -> dict[str, Union[Union[
    DataFrame, dict[str, DataFrame], dict[str, DataFrame], dict[str, DataFrame], dict[str, DataFrame], dict[
        Union[str, Any], Union[DataFrame, Any]]], Any]]:
    """
    返回九个csv格式的数据文件
    :param only_sequence: 是否只给出序列数据
    :return: 九个csv数据文件
    """

    current_path = os.path.dirname(os.path.abspath(__file__))
    # print(current_path)

    def add_abs_path(path: str):
        return os.path.join(current_path, path)

    def sub_read_csv(name: str):
        return pl.read_csv(add_abs_path(name), encoding='gbk')

    def read_many_csv(name: List[str]) -> List[pl.DataFrame]:
        """pl.read_csv 读取后存放到 list 中"""
        store = [pl.read_csv(add_abs_path(n), encoding='gbk') for n in name]
        return store

    # print(__file__)

    name = \
    [
        "广东_成交均价_碳排放权配额(GDEA).csv",
        "期货收盘价(连续)_IPE鹿特丹煤炭.csv",
        "期货收盘价(连续)_NYMEX天然气.csv",
        "期货结算价(连续)_布伦特原油.csv",
        "期货结算价(连续)_欧盟排放配额(EUA).csv",
        "欧元兑美元.csv",
        "沪深300指数.csv",
        "深圳_成交均价_碳排放权(SZA).csv",
        "湖北_成交均价_碳排放权(HBEA).csv"
    ]

    # GDEA, COAL, NG, BRENT, EUA, EURUSD, CSI300, SZA, HBEA = read_many_csv(name)
    Data = read_many_csv(name)
    columns = ["GDEA", "COAL", "NG", "BRENT", "EUA", "EURUSD", "CSI300", "SZA", "HBEA"]

    D = {}
    for i in range(len(columns)):
        D[columns[i]] = Data[i]

    # if filter_weekend:
    #     for df in D.keys():
    #         D[df] = D[df].drop_nulls()

    if only_sequence:
        D["GDEA"] = D["GDEA"][7:-1]
        D["GDEA"] = D["GDEA"].rename \
            (
                {
                    "国家": "时间",
                    "中国": "广东_成交均价_碳排放权配额",
                    "中国_duplicated_0": "广东_收盘价_碳排放权配额",
                    "中国_duplicated_1": "广东_当日成交量_碳排放权配额",
                    "中国_duplicated_2": "广东_累计成交量_碳排放权配额",
                    "中国_duplicated_3": "广东_当日成交额_碳排放权配额",
                    "中国_duplicated_4": "广东_累计成交额_碳排放权配额"
                }
            )

        D["COAL"] = D["COAL"][7:-1]
        D["COAL"] = D["COAL"].rename \
            (
                {
                    "国家": "时间",
                    "英国": "IPE鹿特丹煤炭_期货收盘价_连续",
                    "英国_duplicated_0": "IPE鹿特丹煤炭_期货结算价_连续"
                }
            )

        D["NG"] = D["NG"][7:]
        D["NG"] = D["NG"].rename \
            (
                {
                    "国家": "时间",
                    "美国": "期货收盘价(连续):NYMEX天然气",
                    "美国_duplicated_0": "期货收盘价(活跃合约):NYMEX轻质原油",
                    "美国_duplicated_1": "期货结算价(活跃合约):NYMEX轻质原油",
                    "美国_duplicated_2": "期货收盘价(连续):NYMEX轻质原油"
                }
            )
        temp = D["NG"]
        D["NG"] = \
            {
                # "时间": temp["时间"],
                "NYMEX天然气_期货收盘价_连续": pl.DataFrame([temp["时间"], temp["期货收盘价(连续):NYMEX天然气"]]).drop_nulls(),
                "NYMEX轻质原油_期货收盘价_活跃合约": pl.DataFrame([temp["时间"], temp["期货收盘价(活跃合约):NYMEX轻质原油"]]).drop_nulls(),
                "NYMEX轻质原油_期货结算价_活跃合约": pl.DataFrame([temp["时间"], temp["期货结算价(活跃合约):NYMEX轻质原油"]]).drop_nulls(),
                "NYMEX轻质原油_期货收盘价_连续": pl.DataFrame([temp["时间"], temp["期货收盘价(连续):NYMEX轻质原油"]]).drop_nulls()
            }

        D["BRENT"] = D["BRENT"][7:]
        D["BRENT"] = D["BRENT"].rename \
            (
                {
                    "国家": "时间",
                    "": "期货结算价(连续):布伦特原油",
                    "英国": "全球:现货价:原油(英国布伦特Dtd)",
                    "俄罗斯": "期货收盘价(活跃合约):MICEX 布伦特原油",
                    "俄罗斯_duplicated_0": "期货结算价(活跃合约):MICEX 布伦特原油",
                    "_duplicated_0": "全球:名义商品价格:布伦特原油",
                    "_duplicated_1": "全球:实际市场价格:现货原油:英国布伦特"
                }
            )
        temp = D["BRENT"]
        D["BRENT"] =  \
            {
                # "时间": temp["时间"],
                "布伦特原油_期货结算价_连续": pl.DataFrame([temp["时间"], temp["期货结算价(连续):布伦特原油"]]).drop_nulls(),
                "布伦特原油_全球现货价_活跃合约": pl.DataFrame([temp["时间"], temp["全球:现货价:原油(英国布伦特Dtd)"]]).drop_nulls(),
                "MICEX_布伦特原油_期货收盘价_活跃合约": pl.DataFrame([temp["时间"], temp["期货收盘价(活跃合约):MICEX 布伦特原油"]]).drop_nulls(),
                "MICEX_布伦特原油_期货结算价_活跃合约": pl.DataFrame([temp["时间"], temp["期货结算价(活跃合约):MICEX 布伦特原油"]]).drop_nulls(),
                "布伦特原油_全球名义商品价格": pl.DataFrame([temp["时间"], temp["全球:名义商品价格:布伦特原油"]]).drop_nulls(),
                "布伦特原油_现货_全球实际市场价格": pl.DataFrame([temp["时间"], temp["全球:实际市场价格:现货原油:英国布伦特"]]).drop_nulls()
            }

        D["EUA"] = D["EUA"][7:]
        D["EUA"] = D["EUA"].rename \
            (
                {
                    "国家": "时间",
                    "": "期货结算价(连续):欧盟排放配额(EUA)",
                    "_duplicated_0": "期货成交量(连续):欧盟排放配额(EUA)",
                    "欧盟": "现货结算价:欧盟排放配额(EUA):2021-2030",
                    "_duplicated_1": "期货持仓量(连续):欧盟排放配额(EUA)",
                    "欧盟_duplicated_0": "现货成交量:欧盟排放配额(EUA):2021-2030",
                    "欧盟_duplicated_1": "现货结算价:欧盟航空配额(EUAA):2021-2030",
                    "欧盟_duplicated_2": "欧盟:期货价:碳排放权(EUA)",
                    "欧盟_duplicated_3": "欧盟:现货价:碳排放权(EUA)"
                }
            )
        temp = D["EUA"]
        D["EUA"] = \
            {
                "欧盟排放配额_期货结算价_连续": pl.DataFrame([temp["时间"], temp["期货结算价(连续):欧盟排放配额(EUA)"]]).drop_nulls(),
                "欧盟排放额_期货成交量_连续": pl.DataFrame([temp["时间"], temp["期货成交量(连续):欧盟排放配额(EUA)"]]).drop_nulls(),
                "欧盟排放配额_现货结算价": pl.DataFrame([temp["时间"], temp["现货结算价:欧盟排放配额(EUA):2021-2030"]]).drop_nulls(),
                "欧盟排放配额_期货持仓量_连续": pl.DataFrame([temp["时间"], temp["期货持仓量(连续):欧盟排放配额(EUA)"]]).drop_nulls(),
                "欧盟排放额_现货成交量_2021-2030": pl.DataFrame([temp["时间"], temp["现货成交量:欧盟排放配额(EUA):2021-2030"]]).drop_nulls(),
                "欧盟航空配额_现货结算价_2021-2030": pl.DataFrame([temp["时间"], temp["现货结算价:欧盟航空配额(EUAA):2021-2030"]]).drop_nulls(),
                "欧盟碳排放权_期货价": pl.DataFrame([temp["时间"], temp["欧盟:期货价:碳排放权(EUA)"]]).drop_nulls(),
                "欧盟碳排放权_现货价": pl.DataFrame([temp["时间"], temp["欧盟:现货价:碳排放权(EUA)"]]).drop_nulls()
            }

        D["EURUSD"] = D["EURUSD"][7:]
        D["EURUSD"] = D["EURUSD"].rename \
            (
                {
                    "国家": "时间",
                    "": "欧元兑美元",
                    "中国": "中间价:欧元兑人民币",
                    "中国_duplicated_0": "CFETS:即期汇率:欧元兑人民币",
                    "中国_duplicated_1": "平均汇率:欧元兑人民币",
                    "中国_duplicated_2": "基准汇率:欧元兑人民币"
                }
            )
        temp = D["EURUSD"]
        D["EURUSD"] = \
            {
                "欧元兑美元": pl.DataFrame([temp["时间"], temp["欧元兑美元"]]).drop_nulls(),
                "欧元兑美元_中间价": pl.DataFrame([temp["时间"], temp["中间价:欧元兑人民币"]]).drop_nulls(),
                "欧元兑人民币_即期汇率_CFETS": pl.DataFrame([temp["时间"], temp["CFETS:即期汇率:欧元兑人民币"]]).drop_nulls(),
                "欧元兑人民币_平均汇率": pl.DataFrame([temp["时间"], temp["基准汇率:欧元兑人民币"]]).drop_nulls()
            }

        D["CSI300"] = D["CSI300"][7:]
        D["CSI300"] = D["CSI300"].rename \
            (
                {
                    "国家": "时间",
                    "中国": "沪深300指数",
                    "中国_duplicated_0": "沪深300指数_1",
                    "中国_duplicated_1": "滚动市盈率(TTM):沪深300指数",
                    "中国_duplicated_2": "市盈率:沪深300指数",
                    "中国_duplicated_3": "沪深300指数:换手率",
                    "中国_duplicated_4": "静态市盈率:沪深300指数",
                    "中国_duplicated_5": "沪深300:成交金额",
                    "中国_duplicated_6": "市净率:沪深300指数",
                    "中国_duplicated_7": "中国:沪深300:资金净主动买入额",
                    "中国_duplicated_8": "期货收盘价(活跃合约):沪深300指数期货",
                    "中国_duplicated_9": "沪深300指数:涨跌幅",
                    "中国_duplicated_10": "沪深300:成交量",
                    "中国_duplicated_11": "期货收盘价(连续):沪深300指数期货",
                    "中国_duplicated_12": "期货成交量:沪深300指数期货",
                    "中国_duplicated_13": "期货持仓量:沪深300指数期货",
                    "中国_duplicated_14": "沪深300指数:涨跌点数"
                }
            )
        temp = D["CSI300"]
        t = temp["时间"]
        def f(name: str):
            return pl.DataFrame([t, temp[name]]).drop_nulls()

        D["CSI300"] = \
            (
                {
                    "沪深300指数": f("沪深300指数"),
                    "沪深300指数_TTM": f("滚动市盈率(TTM):沪深300指数"),
                    "沪深300指数_市盈率": f("市盈率:沪深300指数"),
                    "沪深300指数_换手率": f("沪深300指数:换手率"),
                    "沪深300指数_静态市盈率": f("静态市盈率:沪深300指数"),
                    "沪深300指数_成交金额": f("沪深300:成交金额"),
                    "沪深300指数_市净率": f("市净率:沪深300指数"),
                    "沪深300指数_资金净主动买入额": f("中国:沪深300:资金净主动买入额"),
                    "沪深300指数_期货收盘价_活跃合约": f("期货收盘价(活跃合约):沪深300指数期货"),
                    "沪深300指数_涨跌幅": f("沪深300指数:涨跌幅"),
                    "沪深300指数_成交量": f("沪深300:成交量"),
                    "沪深300指数_期货收盘价_连续": f("期货收盘价(连续):沪深300指数期货"),
                    "沪深300指数_期货成交量": f("期货成交量:沪深300指数期货"),
                    "沪深300指数_期货持仓量": f("期货持仓量:沪深300指数期货"),
                    "沪深300指数_涨跌点数": f("沪深300指数:涨跌点数")
                }
            )

        D["SZA"] = D["SZA"][7:].drop_nulls()
        D["SZA"] = D["SZA"].rename \
            (
                {
                    "国家": "时间",
                    "中国": "深圳_SZA市场成交均价_碳排放权",
                    "中国_duplicated_0": "深圳_SZA市场当日成交量_碳排放权",
                    "中国_duplicated_1": "深圳_累计成交量_碳排放权",
                    "中国_duplicated_2": "深圳_累计成交额_碳排放权",
                    "中国_duplicated_3": "深圳_收盘价_碳排放权",
                    "中国_duplicated_4": "深圳_CCER产品当日成交额_碳排放权",
                    "中国_duplicated_5": "深圳_CCER产品当日成交量_碳排放权",
                    "中国_duplicated_6": "深圳_CCER产品成交均价_碳排放权",
                    "中国_duplicated_7": "深圳_SZA市场当日成交额_碳排放权"
                }
            )

        D["HBEA"] = D["HBEA"][7:].drop_nulls()
        D["HBEA"] = D["HBEA"].rename \
            (
                {
                    "国家": "时间",
                    "中国": "湖北_成交均价_碳排放权",
                    "中国_duplicated_0": "湖北_当日成交量_碳排放权",
                    "中国_duplicated_1": "湖北_收盘价_碳排放权",
                    "中国_duplicated_2": "湖北_累计成交量_碳排放权",
                    "中国_duplicated_3": "湖北_累计成交额_碳排放权",
                    "中国_duplicated_4": "湖北_当日成交额_碳排放权"
                }
            )

    # "GDEA", "COAL", "NG", "BRENT", "EUA", "EURUSD", "CSI300", "SZA", "HBEA"

    return D

def store_data \
    (
        path: str=None,
        only_sequence: bool=True
    ) -> None:
    """
    将数据保存。允许给定特定路径
    :param path: 文件夹路径 | 指定路径，默认为当前文件夹内部的 Data 文件夹 | 给定路径尽可能为绝对路径
    :param only_sequence: 
    :return: 
    """

    if path is None:
        path = os.path.dirname(__file__)

        path = os.path.join(path, "Data")
        os.makedirs(path, exist_ok=True)

    D = get_data(only_sequence)

    for key, value in D.items():
        if isinstance(value, pl.DataFrame):
            name = f"{key}.csv"

            temp_path = os.path.join(path, name)

            value.write_csv(temp_path)

        else:
            name_First = key

            name_First_path = os.path.join(path, name_First)
            os.makedirs(name_First_path, exist_ok=True)

            # name_Second_path = os.path

            point: dict[str, pl.DataFrame] = value
            # _key = point.keys()
            # _value: pl.DataFrame = point.values()

            for _key, _value in point.items():

                name_Second = _key
                name = f"{name_Second}.csv"

                temp_path = os.path.join(name_First_path, name)

                _value.write_csv(temp_path)


if __name__ == '__main__':
    store_data()