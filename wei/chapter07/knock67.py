""'''
[description]k-meansクラスタリング
国名に関する単語ベクトルを抽出し，
k-meansクラスタリングをクラスタ数k=5として実行せよ．
'''

import gensim
from sklearn.cluster import KMeans
import numpy as np

filepath = './data/GoogleNews-vectors-negative300.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(filepath, binary=True)

# 単語アナロジーの評価データから、適当な国名リストを取得
countries = set()
with open('./knock64_add_sim.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.split()
        if line[0] in ['capital-common-countries', 'capital-world']:
            countries.add(line[2])
        elif line[0] in ['currency', 'gram6-nationality-adjective']:
            countries.add(line[1])
countries = list(countries)

# 単語ベクトルを取得
countries_vec = [model[country] for country in countries]
# K平均クラスタリング
kmeans = KMeans(n_clusters=5)
kmeans.fit(countries_vec)
for i in range(5):
    cluster = np.where(kmeans.labels_ == i)[0]
    print('cluster', i)
    print(','.join([countries[k] for k in cluster]))

'''
cluster 0
Serbia,Belarus,Poland,Azerbaijan,Slovakia,Lithuania,Ukraine,Moldova,Malta,Macedonia,Kazakhstan,Uzbekistan,Georgia,Latvia,Greece,Montenegro,Romania,Cyprus,Kyrgyzstan,Albania,Estonia,Slovenia,Russia,Bulgaria,Armenia,Hungary,Croatia
cluster 1
Zambia,Botswana,Burundi,Nigeria,Liberia,Algeria,Namibia,Malawi,Mauritania,Somalia,Tunisia,Rwanda,Gabon,Mozambique,Uganda,Ghana,Madagascar,Sudan,Angola,Zimbabwe,Kenya,Gambia,Guinea,Niger,Senegal,Mali,Eritrea
cluster 2
Libya,Afghanistan,Indonesia,Morocco,Iran,Israel,Iraq,Jordan,India,Tajikistan,Korea,Cambodia,Bhutan,Turkey,Taiwan,China,Bahrain,Lebanon,Vietnam,Syria,Qatar,Turkmenistan,Laos,Nepal,Oman,Malaysia,Bangladesh,Thailand,Pakistan,Egypt
cluster 3
Brazil,Uruguay,Peru,Philippines,Guyana,Bahamas,Venezuela,Argentina,Jamaica,Cuba,Fiji,Samoa,Suriname,Colombia,Honduras,Ecuador,Nicaragua,Dominica,Belize,Mexico,Chile
cluster 4
England,USA,Switzerland,Greenland,France,Netherlands,Italy,Sweden,Spain,Liechtenstein,Norway,Austria,Japan,Finland,Ireland,Portugal,Europe,Canada,Denmark,Germany,Iceland,Belgium,Tuvalu,Australia
'''
