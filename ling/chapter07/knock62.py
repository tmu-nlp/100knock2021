from gensim.models import KeyedVectors
model=KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz',binary=True)
for i in model.most_similar('United_States', topn=10):
    print(i)