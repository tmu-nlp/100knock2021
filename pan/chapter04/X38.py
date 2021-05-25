#単語の出現頻度のヒストグラムを描け．ただし，横軸は出現頻度を表し，1から単語の出現頻度の最大値までの線形目盛とする．縦軸はx軸で示される出現頻度となった単語の異なり数（種類数）である.
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from X30 import mecab
from collections import Counter

def describe(table)->'None':
    sample = []
    data = []
    print(table)
    """
    for key, value in table:
        sample.append(key)
        data.append(value)
    """
    for i in range(20):
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
    dic = word_count()
    re = []
    i = 0
    for key in dic:
        re.append(dic[key])
        i += 1
    describe((Counter(re)).most_common(i))
