from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

simularities = model.most_similar(positive=['Spain', 'Athens'], negative=['Madrid'])

print('{:20s}{}'.format('単語','類似度'))

for rank in simularities[:10]:
    print('{:20s}{}'.format(rank[0], rank[1]))

'''

単語                  類似度
Greece              0.689848005771637
Aristeidis_Grigoriadis0.560684859752655
Ioannis_Drymonakos  0.5552910566329956
Greeks              0.5450688004493713
Ioannis_Christou    0.5400862693786621
Hrysopiyi_Devetzi   0.5248443484306335
Heraklio            0.5207761526107788
Athens_Greece       0.516880989074707
Lithuania           0.5166865587234497
Iraklion            0.5146791934967041

'''