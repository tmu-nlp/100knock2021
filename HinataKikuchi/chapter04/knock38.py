from knock35 import sorted_nouns
import matplotlib.pyplot as plt

val_ = []
key_ = []

for noun in sorted_nouns:
	key_.append(noun[0])
	val_.append(noun[1])
plt.hist(val_[:10], bins=10, orientation="horizontal")
plt.savefig('./knock38.jpg')