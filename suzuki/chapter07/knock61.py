from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

print(model.similarity("United_States", "U.S."))

'''

0.73107743

'''