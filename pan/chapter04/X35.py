#文章中に出現する単語とその出現頻度を求め，出現頻度の高い順に並べよ．

from X30 import mecab
from collections import Counter

def word_count():
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
    print(word_count().most_common(10))
