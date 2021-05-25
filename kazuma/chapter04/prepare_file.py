import urllib.request
import MeCab


url='https://nlp100.github.io/data/neko.txt'
save_name='data/neko.txt'

urllib.request.urlretrieve(url, save_name)

mecab = MeCab.Tagger ('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')

with open ("data/neko.txt", "r") as f1,\
     open ("data/neko.txt.mecab", "w") as f2:
    for line in f1:
        if line.strip() == "":
            continue
        t = MeCab.Tagger('')
        f2.write(t.parse(line))
  