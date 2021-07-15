""'''
80. ID番号への変換
最も頻出する単語に1，2番目に頻出する単語に2…のように、学習データ中で2回以上出現する単語にID番号を付与せよ.
与えられた単語列に対して，ID番号の列を返す関数を実装せよ．
ただし，出現頻度が2回未満の単語のID番号はすべて0とせよ.
第6章の問題50で作成したデータを利用
学習データの単語をカウントし、2回以上登場するものをキーとして,頻度順位IDを登録
'''


from collections import defaultdict
import string
import pandas as pd
from sklearn.model_selection import train_test_split


# 記号をスペースに置換するtable
table = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

# データの準備
infile = '../chapter06/data/NewsAggregatorDataset/newsCorpora_re.csv'
df = pd.read_csv(infile, header=None, sep='\t',
                 names=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])
df = df.loc[df['PUBLISHER'].isin(
['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail']), ['TITLE', 'CATEGORY']]
train, valid_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=123, stratify=df['CATEGORY'])
valid, test = train_test_split(valid_test, test_size=0.5, shuffle=True, random_state=123, stratify=valid_test['CATEGORY'])

def get_word2id():
    # 単語の頻度集計
    d = defaultdict(int)
    for text in train['TITLE']:
        for word in text.translate(table).split():
            d[word] += 1
    print(d.items())        # -> [('REFILE', 59), ('UPDATE', 1046), ...]
# 単語の出現頻度の大きい順でソートしたリストを作成(最も出現する単語が先頭に来る)
    d = sorted(d.items(), key=lambda x: x[1], reverse=True)

# 最も頻出する単語に1という方法で単語にID番号を付与し辞書を作成。
# ただし、出現頻度が2回以上の単語を登録
    word2id = {word: i + 1 for i, (word, cnt) in enumerate(d) if cnt > 1}
    return word2id


# 辞書を用いて与えられた単語列をID番号の列に変換する関数を定義
def tokenizer(text, word2id, unk=0):
    res = []
    # 記号をスペースに置換,スペースで分割したID列に変換(辞書になければunkで0を返す)
    table = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    for word in text.translate(table).split():
        res.append(word2id.get(word, unk))
    return res


if __name__ == '__main__':
    word2id = get_word2id()
    print(f'ID数:{len(set(word2id.values()))}\n')        # ->ID数:9405
    print('---頻度上位20語---')
    for key in list(word2id)[:20]:
        print(f'{key}:{word2id[key]}')

    text = train.iloc[2, train.columns.get_loc('TITLE')]   # TITLE列索引为2的行
    print(f'text:{text}')
    print(f'ID列:{tokenizer(text, word2id)}')
    '''
    text:Kids Still Get Codeine In Emergency Rooms Despite Risky Side Effects (STUDY)
    ID列:[540, 321, 236, 0, 16, 3528, 0, 1238, 6668, 4200, 2664, 2335]'''

