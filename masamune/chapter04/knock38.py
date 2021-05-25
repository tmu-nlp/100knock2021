import ast
from collections import defaultdict
import matplotlib.pyplot as plt

word_cnt = defaultdict(lambda: 0)
with open('morpheme.txt') as file:
    for sentence in file:
        sentence_dic = ast.literal_eval(sentence) #文字列を辞書に変換
        for morpheme in sentence_dic:
            if morpheme['pos'] != '記号':
                word_cnt[morpheme['base']] += 1

    word_cnt = sorted(word_cnt.items(), reverse=True, key = lambda x: x[1])

    x = [word_cnt[i][1] for i in range(len(word_cnt))]
    plt.xlabel('freqency')
    plt.ylabel('word count')
    plt.hist(x, bins=100)
    plt.show()