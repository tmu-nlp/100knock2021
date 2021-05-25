import collections
from knock30 import analized_dicts as d
import MeCab
import re
import matplotlib.pyplot as plt

text_path = './neko.txt'
m = MeCab.Tagger('-d /etc/alternatives/mecab-dictionary')

word_list = []
with open(text_path) as f:
	for text in f.read().split('\n'):
		if '猫' in text :
			for line in m.parse(text).split('\n'):
				if line != 'EOS' and line != '' and '猫\t' not in re.findall('.*\t', line):
					word_list.append(re.findall('.*\t', line)[0].replace('\t',''))
text_with_cat = collections.Counter(word_list).most_common()
val_ = []
key_ = []
for word in text_with_cat[:10]:
	print(word)
	key_.append(word[0])
	val_.append(word[1])

plt.bar(key_, val_)
plt.savefig('knock37.jpg')

###これのがいいかも？###
#https://qiita.com/yniji/items/3fac25c2ffa316990d0c
#mac ですが
# plt.rcParams["font.family"] = "Hiragino sans"
# を入れたらできました というご意見も