from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from stemming.porter2 import stem
import pandas as pd
import joblib
import re

def preprocessing(title):
    title = title.lower()
    title = re.sub(r"\d+", "0", title)
    title = " ".join(word_tokenize(title))
    title = stem(title)

    return title

#データ読み込み
names = ['TITLE', 'CATEGORY']
train_df = pd.read_csv('./data/train.txt', sep='\t', header=None, names=names)
valid_df = pd.read_csv('./data/valid.txt', sep='\t', header=None, names=names)
test_df = pd.read_csv('./data/test.txt', sep='\t', header=None, names=names)

#前処理
train_df['TITLE'] = train_df['TITLE'].apply(preprocessing)
valid_df['TITLE'] = valid_df['TITLE'].apply(preprocessing)
test_df['TITLE'] = test_df['TITLE'].apply(preprocessing)

#文をベクトル化
vectorizer = TfidfVectorizer(use_idf=True, norm="l2", smooth_idf=True)
X_train = vectorizer.fit_transform(train_df['TITLE'])
X_valid = vectorizer.transform(valid_df['TITLE'])
X_test = vectorizer.transform(test_df['TITLE'])

#データ保存
X_train = pd.DataFrame(X_train.toarray())
X_valid = pd.DataFrame(X_valid.toarray())
X_test = pd.DataFrame(X_test.toarray())
X_train.to_csv('./data/train.feature.txt', sep='\t', index=False, header=None)
X_valid.to_csv('./data/valid.feature.txt', sep='\t', index=False, header=None)
X_test.to_csv('./data/test.feature.txt', sep='\t', index=False, header=None)

joblib.dump(vectorizer.vocabulary_, 'vocabulary.joblib')