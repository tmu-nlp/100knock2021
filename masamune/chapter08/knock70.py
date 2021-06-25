from gensim.models import KeyedVectors
import pandas as pd
import numpy as np

#学習済み単語ベクトル(knock60)
model = KeyedVectors.load_word2vec_format('../chapter07/GoogleNews-vectors-negative300.bin.gz', binary=True)

#データ読み込み
train = pd.read_csv('../chapter06/data/train.txt', sep='\t', header=None)
valid = pd.read_csv('../chapter06/data/valid.txt', sep='\t', header=None)
test = pd.read_csv('../chapter06/data/valid.txt', sep='\t', header=None)

#ベクトル化
cat2num = str.maketrans({'b': 0, 't': 1, 'e': 2, 'm': 3})
def vectorize(df):
    #TITLE
    vecs = []
    for title in df.iloc[:, 0]:
        vec = []
        for word in title.split():
            if word in model:
                vec.append(model[word])
        if len(vec) == 0:
            vec = np.zeros(300)
        else:
            vec = np.average(vec, axis=0)
        
        vecs.append(vec)
        
    #CATEGORY
    label = []
    cat2num = {'b': 0, 't': 1, 'e': 2, 'm': 3}
    for category in df.iloc[:, 1]:
        label.append(cat2num[category])
    
    return pd.DataFrame(vecs, index=None), pd.DataFrame(label, index=None)

if __name__ == '__main__':
    X_train, y_train = vectorize(train)
    X_valid, y_valid = vectorize(valid)
    X_test, y_test = vectorize(test)

    X_train.to_csv('./data/X_train.txt', header=None, index=None)
    y_train.to_csv('./data/y_train.txt', header=None, index=None)
    X_valid.to_csv('./data/X_valid.txt', header=None, index=None)
    y_valid.to_csv('./data/y_valid.txt', header=None, index=None)
    X_test.to_csv('./data/X_test.txt', header=None, index=None)
    y_test.to_csv('./data/y_test.txt', header=None, index=None)