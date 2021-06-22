""'''
[description]加法構成性によるアナロジー
“Spain”の単語ベクトルから”Madrid”のベクトルを引き，”Athens”のベクトルを足したベクトルを計算し，
そのベクトルと類似度の高い10語とその類似度を出力せよ．
'''
import gensim

filepath = './data/GoogleNews-vectors-negative300.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(filepath, binary=True)


vec = model['Spain'] - model['Madrid'] + model['Athens']
res_tmp = model.most_similar(positive=['Spain', 'Athens'], negative=['Madrid'], topn=10)
result = dict()
for elem in res_tmp:
    key = elem[0]
    result[key] = elem[1]
print(result)

'''
{'Greece': 0.6898480653762817, 'Aristeidis_Grigoriadis': 0.5606849193572998, 'Ioannis_Drymonakos': 0.555290937423706, 'Greeks': 0.5450686812400818, 'Ioannis_Christou': 0.5400862693786621, 'Hrysopiyi_Devetzi': 0.5248445272445679, 'Heraklio': 0.5207759141921997, 'Athens_Greece': 0.516880989074707, 'Lithuania': 0.5166865587234497, 'Iraklion': 0.5146791338920593}
'''
