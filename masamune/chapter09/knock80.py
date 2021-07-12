import pandas as pd
from collections import defaultdict
import pickle

#データ読み込み
names = ['TITLE', 'CATEGORY']
train_df = pd.read_csv('../chapter06/data/train.txt', sep='\t', header=None, names=names)

#辞書作成
dic = defaultdict(lambda: 0)
for line in train_df['TITLE']:
    words = line.split()
    for word in words:
        dic[word.lower()] += 1
dic = sorted(dic.items(), key=lambda x:x[1], reverse=True)

#単語とidの辞書
word2id = {}
for i, dic in enumerate(dic):
    word = dic[0]
    cnt = dic[1]
    if cnt < 2:
        word2id[word] = 0
    else:
        word2id[word] = i+1
        
def give_id(sentence, word2id):
    words = sentence.split()
    ids = [word2id[word.lower()] for word in words]
    return ids

with open('word2id.pkl', 'wb') as f:
    pickle.dump(word2id, f)

if __name__ == '__main__':
    for i, sentence in enumerate(train_df['TITLE']):
        print(f'単語列：{sentence}')
        print(f'ID番号：{give_id(sentence, word2id)}')
        if i == 2:
            break

'''
単語列：Bouygues confirms improved offer for Vivendi's SFR
ID番号：[3619, 609, 2063, 212, 6, 5882, 3050]
単語列：US stocks end off highs, Dow up for week; gold at six-week lows
ID番号：[12, 28, 194, 53, 2326, 1412, 21, 6, 0, 228, 15, 4493, 732]
単語列：'Chef' Director: The Creative Driving Force Behind The Food Culture Is Latino  ...
ID番号：[3620, 5883, 4, 5884, 815, 1536, 555, 4, 644, 2649, 17, 0, 2]
'''