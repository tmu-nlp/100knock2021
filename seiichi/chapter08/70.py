import pandas as pd
import gensim
import numpy as np

train = pd.read_csv('../chapter06/data/train.txt',sep='\t',header=None)
valid = pd.read_csv('../chapter06/data/valid.txt',sep='\t',header=None)
test = pd.read_csv('../chapter06/data/test.txt',sep='\t',header=None)
model = gensim.models.KeyedVectors.load_word2vec_format('../chapter07/data/GoogleNews-vectors-negative300.bin', binary=True)

d = {'b':0, 't':1, 'e':2, 'm':3}
y_train = train.iloc[:,0].replace(d)
y_train.to_csv('./data/y_train.txt',header=False, index=False)
y_valid = valid.iloc[:,0].replace(d)
y_valid.to_csv('./data/y_valid.txt',header=False, index=False)
y_test = test.iloc[:,0].replace(d)
y_test.to_csv('./data/y_test.txt',header=False, index=False)

def write_X(file_name, df):
    with open(file_name,'w') as f:
        for text in df.iloc[:,1]:
            vectors = []
            for word in text.split():
                if word in model.vocab:
                    vectors.append(model[word])
            if (len(vectors)==0):
                vector = np.zeros(300)
            else:
                vectors = np.array(vectors)
                vector = vectors.mean(axis=0)
            vector = vector.astype(np.str).tolist()
            output = ' '.join(vector)+'\n'
            f.write(output)
write_X('./data/X_train.txt', train)
write_X('./data/X_valid.txt', valid)
write_X('./data/X_test.txt', test)