import MeCab
import sys

text_path = './neko.txt'
analized_text_path = './neko.txt.mecab'
m = MeCab.Tagger('-d /etc/alternatives/mecab-dictionary')

with open(text_path) as input, open(analized_text_path, mode='w') as output:
	output.write(m.parse(input.read()))

