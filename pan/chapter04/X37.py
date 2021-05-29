#「猫」とよく共起する（共起頻度が高い）10語とその出現頻度をグラフ（例えば棒グラフなど）で表示せよ．

import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from X30 import mecab
from collections import Counter

def co_occ(target)->list:
    lines = mecab(file)
    co_list = []
    nouns = []
    tar_sen = []
    find_tar = 0
    for line in lines:
        for word in line:
            if word['surface'] == '猫': find_tar = 1
            else: nouns.append(word['surface'])
            if word['surface'] == "。":
                if find_tar == 1:
                    for noun in nouns:
                        co_list.append(noun)
                nouns = []
                find_tar = 0
    return(co_list)

def describe_top10(table)->'None':
    sample = []
    data = []
    for i in range(10):
        sample.append(table[i][0])
        data.append(table[i][1])
    left = np.array(sample)
    height = np.array(data)
    plt.bar(left, height)
    plt.show()

if __name__ == '__main__':
    file = '/users/kcnco/github/100knock2021/pan/chapter04/neko.txt.mecab'
    t = 0
    #上位10個
    re = Counter(co_occ("cat")).most_common(30)
    describe_top10(describe_top10(re))
