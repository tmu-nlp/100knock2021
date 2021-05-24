from knock30 import load_mecab
from collections import Counter
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

word_count = Counter()
for line in load_mecab():
    if '猫' in [d.get('surface') for d in line]: 
        for morpheme in line:
            if morpheme['surface'] != '猫':
                word_count[morpheme['surface']]+=1

left, height = [], []
for word in word_count.most_common(10):
    left.append(word[0]), height.append(word[1])
plt.bar(left, height)
plt.title('「猫」と共起頻度の高い上位10語')
plt.xlabel('単語')
plt.ylabel('共起頻度')
plt.show()