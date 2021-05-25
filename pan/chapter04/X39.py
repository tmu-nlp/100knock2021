#単語の出現頻度順位を横軸，その出現頻度を縦軸として，両対数グラフをプロットせよ．
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from X30 import mecab
from collections import Counter

def describe(table)->'None':
    val = []
    for key, value in table:
        val.append(value)
    
    plt.scatter(
        range(1, len(val) + 1),val)

    plt.xlim(1, len(val) + 1)
    plt.ylim(1, val[0])

    plt.xscale('log')
    plt.yscale('log')

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
    dic = word_count()
    re = []
    for key in dic:
        t += 1
    describe(dic.most_common(t))
