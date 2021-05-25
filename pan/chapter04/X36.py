#####
#出現頻度が高い10語とその出現頻度をグラフ（例えば棒グラフなど）で表示せよ．

import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from X30 import mecab
from collections import Counter

def describe_top10(table)->"None":
    sample = []
    data = []
    for i in range(10):
        sample.append(table[i][0])
        data.append(table[i][1])
    left = np.array(sample)
    height = np.array(data)
    plt.bar(left, height)
    plt.show()

def word_count()->list:
    lines = mecab(file)
    longest_N_list = []
    word_list = []
    for line in lines:
        for word in line:
            word_list.append(word['surface'])
    return(Counter(word_list))

if __name__ == '__main__':
    file = '/users/kcnco/github/100knock2021/pan/chapter04/neko.txt.mecab'
    t = 0
    #上位10個
    describe_top10(word_count().most_common(10))
