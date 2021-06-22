!pip install stemming

path = '/content/drive/MyDrive/nlp100/'
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from stemming.porter2 import stem
import pickle
train_df = pd.read_table(path + 'train.txt')
valid_df = pd.read_table(path + 'valid.txt')
test_df = pd.read_table(path + 'test.txt')
#データ抜き出し
X_train = train_df['TITLE']
X_valid = valid_df['TITLE']
X_test = test_df['TITLE']

def preprocess(text):
    #小文字化、数字を0に, ステミング
    text = text.lower()
    text = stem(text)
    text = re.sub('\d+', '0', text)
    
    return text

#前処理
x_train = X_train.map(lambda x: preprocess(x))
X_valid = X_valid.map(lambda x: preprocess(x))
X_test = X_test.map(lambda x: preprocess(x))

vectorizer = TfidfVectorizer(min_df=10, smooth_idf=True)
#train　のみ特徴量とする
X_train = vectorizer.fit_transform(X_train)
X_valid = vectorizer.transform(X_valid)
X_test = vectorizer.transform(X_test)



#csv で保存
X_train = pd.DataFrame(X_train.toarray(), columns=vectorizer.get_feature_names())
X_valid = pd.DataFrame(X_valid.toarray(), columns=vectorizer.get_feature_names())
X_test = pd.DataFrame(X_test.toarray(), columns=vectorizer.get_feature_names())

X_train.to_csv(path + 'train_feature.csv', sep = "\t", index = False)
X_valid.to_csv(path + 'valid_feature.csv', sep = "\t", index = False)
X_test.to_csv(path + 'test_feature.csv', sep = "\t", index = False)

#vocabulary
filename = path + 'vocabulary.sav'
pickle.dump(vectorizer.vocabulary_, open(filename, 'wb'))