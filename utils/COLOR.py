import colorama as cm
from typing import Literal, Any, Union
# from _typeshed import SupportsWrite
import enum


__all__ = ["printc", "Reset"]

cm.init(autoreset=True)

Fore = cm.Fore

class Color():
    black = Fore.BLACK
    red = Fore.RED
    green = Fore.GREEN
    yellow = Fore.YELLOW
    blue = Fore.BLUE
    magenta = Fore.MAGENTA  # 品红
    cyan = Fore.CYAN  # 青色
    white = Fore.WHITE


Reset = cm.Style.RESET_ALL

def printc(*values, color: Literal["black", "red", "green", "yellow", "blue", "magenta", "cyan", "white"] = None, sep: str = ' ', end: str = '\n') -> None:
    """
    支持多參數、自定義分隔符和結束符的彩色打印。
    """
    if color is None:
        print(*values, sep=sep, end=end)
        return

    c = getattr(Color, color, Color.white)
    text = sep.join(str(arg) for arg in values)
    print(c + text, end=end)

if __name__ == '__main__':
    print(Color.yellow + "hello world")

"""
    BLACK           = 30
    RED             = 31
    GREEN           = 32
    YELLOW          = 33
    BLUE            = 34
    MAGENTA         = 35
    CYAN            = 36
    WHITE
"""