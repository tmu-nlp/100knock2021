from knock30 import get_nekoList
from collections import defaultdict
from pprint import pprint

d1 = defaultdict(lambda:0)
neko = get_nekoList()
str1 = ""
for line in neko:
    flag = False  # 一回目の名詞かどうか
    flag2 = False # 連接かどうか（つまり、d1に加えるかどうか）
    for mor in line:
        if flag :
            if mor["pos"]=="名詞":
                str1 += mor["surface"]
                flag2 = True
            else:
                if flag2:
                    d1[str1] += 1
                flag = False
                flag2 = False
        elif mor["pos"]=="名詞":
            str1 = mor["surface"]
            flag = True
print(d1)