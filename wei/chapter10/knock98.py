# データ準備


import re
import sentencepiece as spm
import jieba
import pandas as pd
from sklearn.model_selection import train_test_split
import spacy


# ソース言語とターゲット言語をそれぞれ抽出、保存
filepath = './zh-ja/zh-ja.bicleaner05.txt'
file = open(filepath, 'r', encoding='utf-8')
data = [x.strip().split('\t') for x in file.readlines()]
data = [x for x in data if len(x) == 4]
data = [[x[3], x[2]] for x in data]


with open('jparacrawl.ja', 'w', encoding='utf-8') as f, open('jparacrawl.zh', 'w', encoding='utf-8') as g:
    for ja, zh in data:
        print(ja, file=f)
        print(zh, file=g)

# サブワード化
spm.SentencePieceTrainer.Train('--input=kftt-data-1.0/data/orig/kyoto-train.ja --model_prefix=kyoto_ja --vocab_size=16000 --character_coverage=1.0')
sp = spm.SentencePieceProcessor()
sp.Load('kyoto_ja.model')
with open('jparacrawl.ja','r', encoding='utf-8') as f, \
    open('jparacrawl_sub.ja', 'w', encoding='utf-8') as g:
    for x in f:
        x = x.strip()
        #x = re.sub(r'\s+', ' ', x)
        x = sp.encode_as_pieces(x)
        x = ' '.join(x)
        print(x, file=g)


with open('jparacrawl.zh', 'r', encoding='utf-8') as f, \
    open('jparacrawl_sub.zh', 'w', encoding='utf-8') as g:
    for x in f:
        x = x.strip()
        #x = re.sub(r'\s+', ' ', x)
        x = jieba.lcut(x)
        x = ' '.join(x)
        print(x, file=g)

df_ja = pd.read_csv('jparacrawl_sub.ja', header=None, encoding='utf-8')
df_zh = pd.read_csv('jparacrawl_sub.zh', header=None, encoding='utf-8')


train_ja, valid_test_ja = train_test_split(df_ja, test_size=0.2, shuffle=True, random_state=123)
valid_ja, test_ja = train_test_split(valid_test_ja, test_size=0.5, shuffle=True, random_state=123)
train_zh, valid_test_zh = train_test_split(df_zh, test_size=0.2, shuffle=True, random_state=123)
valid_zh, test_zh = train_test_split(valid_test_zh, test_size=0.5, shuffle=True, random_state=123)

train_ja.to_csv('train.sub.ja', header=False)
valid_ja.to_csv('valid.sub.ja', header=False)
test_ja.to_csv('test.sub.ja', header=False)
train_zh.to_csv('train.sub.zh', header=False)
valid_zh.to_csv('valid.sub.zh', header=False)
test_zh.to_csv('test.sub.zh', header=False)




