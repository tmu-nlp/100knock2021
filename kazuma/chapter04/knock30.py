# 表層形\t品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用型,活用形,原形,読み,発音
# ↑MeCabの出力形式
from pprint import pprint

def get_nekoList():
    with open("data/neko.txt.mecab", "r") as f:
        list1 = []
        for line in f:
            dict1 = {}
            mor = line.strip("\n") # mor stands for morpheme
            if mor == "EOS":
                yield list1
                list1 = []
                continue
            mor = mor.split("\t")
            dict1["surface"] = mor[0]
            mor_info = mor[1].split(",")
            dict1["base"] = mor_info[6]
            dict1["pos"] = mor_info[0]
            dict1["pos1"] = mor_info[1]
            list1.append(dict1)

if __name__ == "__main__":
    neko = get_nekoList()

    for i,line in enumerate(neko):
        pprint(line)
        if i == 5:
            break