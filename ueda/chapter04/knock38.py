from knock35 import word_counter
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

word_count = word_counter(None)
plt.hist([i[1] for i in word_count], bins=200)
plt.title('単語の出現頻度のヒストグラム')
plt.xlabel('出現頻度')
plt.ylabel('単語の異なり数')
plt.show()

