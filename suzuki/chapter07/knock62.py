from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

simularities = model.most_similar("United_States")

print('{:20s}{}'.format('単語','類似度'))

for rank in simularities[:10]:
    print('{:20s}{}'.format(rank[0], rank[1]))

'''

単語                  類似度
Unites_States       0.7877248525619507
Untied_States       0.754136860370636
United_Sates        0.7400723099708557
U.S.                0.7310770750045776
theUnited_States    0.6404393315315247
America             0.6178407669067383
UnitedStates        0.6167312264442444
Europe              0.6132986545562744
countries           0.6044804453849792
Canada              0.6019068360328674

'''