""'''
[description]Ward法によるクラスタリング
国名に関する単語ベクトルに対し，Ward法による階層型クラスタリングを実行せよ．
さらに，クラスタリング結果をデンドログラムとして可視化せよ
'''
import gensim
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage



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
plt.figure(figsize=(15, 5))
Z = linkage(countries_vec, method='ward')
dendrogram(Z, labels=countries)
plt.show()
