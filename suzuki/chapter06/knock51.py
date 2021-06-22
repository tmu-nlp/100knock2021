import re
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

#前処理
def preprocessing(text):
    #記号を削除
    text = text.translate(str.maketrans( '', '', string.punctuation))

    # 小文字化
    text = text.lower()

    # 数字列を0に置換
    text = re.sub('[0-9]+', '0', text)

    return text

#txtをデータフレーム化
train = pd.read_table('train.txt', header = None, sep = '\t', names = ['CATEGORY', 'TITLE'])
test = pd.read_table('test.txt', header = None, sep = '\t', names = ['CATEGORY', 'TITLE'])
valid = pd.read_table('valid.txt', header = None, sep = '\t', names = ['CATEGORY', 'TITLE'])

#TfidfVectorizer
vec = TfidfVectorizer(min_df=5, ngram_range=(1, 2))

#前処理
train['TITLE'] = train['TITLE'].map(lambda x: preprocessing(x))
test['TITLE'] = test['TITLE'].map(lambda x: preprocessing(x))
valid['TITLE'] = test['TITLE'].map(lambda x: preprocessing(x))

#ベクトル化
X_train = vec.fit_transform(train['TITLE'])
X_test = vec.transform(test['TITLE'])
X_valid = vec.transform(valid['TITLE'])

#データフレームに変換
X_train = pd.DataFrame(X_train.toarray(), columns=vec.get_feature_names())
X_test = pd.DataFrame(X_test.toarray(), columns=vec.get_feature_names())
X_valid = pd.DataFrame(X_valid.toarray(), columns=vec.get_feature_names())

#書き出し
X_train.to_csv('train.feature.txt', sep='\t', index=False)
X_test.to_csv('test.feature.txt', sep='\t', index=False)
X_valid.to_csv('valid.feature.txt', sep='\t', index=False)


'''

print(X_train.head())

   0amazon  0american  0apple  0argentina  0bank  0bank of  0blackberry  0bn  0bnp  ...  zakis review  zealand  zendaya  zero  zone  zone bond  zone bonds   zs  zuckerberg
0      0.0        0.0     0.0         0.0    0.0       0.0          0.0  0.0   0.0  ...           0.0      0.0      0.0   0.0   0.0        0.0         0.0  0.0         0.0
1      0.0        0.0     0.0         0.0    0.0       0.0          0.0  0.0   0.0  ...           0.0      0.0      0.0   0.0   0.0        0.0         0.0  0.0         0.0
2      0.0        0.0     0.0         0.0    0.0       0.0          0.0  0.0   0.0  ...           0.0      0.0      0.0   0.0   0.0        0.0         0.0  0.0         0.0
3      0.0        0.0     0.0         0.0    0.0       0.0          0.0  0.0   0.0  ...           0.0      0.0      0.0   0.0   0.0        0.0         0.0  0.0         0.0
4      0.0        0.0     0.0         0.0    0.0       0.0          0.0  0.0   0.0  ...           0.0      0.0      0.0   0.0   0.0        0.0         0.0  0.0         0.0

[5 rows x 5524 columns]

'''