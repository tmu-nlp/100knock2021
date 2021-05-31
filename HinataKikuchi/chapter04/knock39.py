from knock35 import sorted_nouns
import matplotlib.pyplot as plt

key_=[]
val_=[]
for noun in sorted_nouns:
	key_.append(noun[0])
	val_.append(noun[1])

plt.plot(key_, val_)
plt.xscale('log')
plt.yscale('log')
plt.savefig('./knock39.jpg')

###ANS###
# zipで書いてあげるとl.6からのfor文はよりきれいになる。
# 一度全部格納したものを、dict列として保存しておくと良い。
# もちろんジェネレータでもよき。練習になる