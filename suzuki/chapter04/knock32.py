#動詞の基本形をすべて抽出せよ．

from knock30 import make_morpheme

f = open('neko.txt.mecab', 'r')

m = make_morpheme(f)

for i in m:
    if i['pos'] == '動詞':
        print(i['base'])

f.close()