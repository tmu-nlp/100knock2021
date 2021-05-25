import ast
from collections import defaultdict
import matplotlib.pyplot as plt
import japanize_matplotlib

word_cnt = defaultdict(lambda: 0)
with open('morpheme.txt') as file:
    for sentence in file:
        sentence_dic = ast.literal_eval(sentence) #文字列を辞書に変換
        for morpheme in sentence_dic:
            if morpheme['pos'] != '記号':
                word_cnt[morpheme['base']] += 1

    word_cnt = sorted(word_cnt.items(), reverse=True, key = lambda x: x[1])

    for i in range(10):
        print(word_cnt[i])