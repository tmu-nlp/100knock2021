import matplotlib.pyplot as plt
from knock35 import sorted_nouns
from matplotlib.font_manager import FontProperties

###TO_MAKE_IT_JAPANESE###
fp = FontProperties(fname='/home/hkikuchi/100knock2021/HinataKikuchi/chapter04/.venv/lib/python3.8/site-packages/matplotlib/mpl-data/fonts/ttf/ipaexg.ttf')
plt.rcParams['font.family'] = 'IPAPGothic'
#########################

val_ = []
key_ = []
for noun in sorted_nouns[:10]:
	key_.append(noun[0])
	val_.append(noun[1])

plt.title('日本語化できないですーどうしたらいいのー？',fontproperties=fp)
plt.bar(key_, val_)
plt.savefig('knock36.jpg')
