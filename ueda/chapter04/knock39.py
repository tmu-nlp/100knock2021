from knock35 import word_counter
import matplotlib.pyplot as plt
from collections import Counter
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

word_count = word_counter(None)
plt.scatter(list(range(len(word_count))), [i[1] for i in word_count])
plt.xscale('log')
plt.yscale('log') 
plt.title('Zipfの法則')
plt.xlabel('出現頻度順位')
plt.ylabel('出現頻度')
plt.show()

