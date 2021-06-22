from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz', binary=True)

similarity = model.similarity("United_States", "U.S.") #コサイン類似度
print(similarity)

'''
出力結果
0.73107743
'''