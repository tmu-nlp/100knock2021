from knock35 import word_counter
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

word_count = word_counter(10)
left, height = [], []
for word in word_count:
    left.append(word[0]), height.append(word[1])
plt.bar(left, height)
plt.title('頻度上位10語')
plt.xlabel('単語')
plt.ylabel('出現頻度')
plt.show()

