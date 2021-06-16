import gensim
import pickle
file ='GoogleNews-vectors-negative300.bin.gz'
model = gensim.models.KeyedVectors.load_word2vec_format(file, binary=True)
print(model['United_States'])

###シリアライズ
#filename = path + 'model.sav'
#pickle.dump(model, open(filename, 'wb'))