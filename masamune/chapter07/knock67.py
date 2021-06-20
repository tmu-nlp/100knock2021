from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
import pickle

model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz', binary=True)

#国名を抽出
countries = set()
with open('./questions-words-re.txt') as f:
    flg = None
    for line in f:
        line = line.split()
        if len(line) == 1:
            if 'capital' in line[0]: #カテゴリーにcapitalが入っていると国名
                flg = True 
            else:             
                flg = False
            continue

        if flg:
            #国名を追加
            countries.add(line[1])
            countries.add(line[3])

#ベクトルの作成
countries_vec = []
for country in countries:
    countries_vec.append(model[country])

#k-meansインスタンス
kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(countries_vec)

#辞書作成
dic = {}
for i in range(5):
    dic[f'cluster{i}'] = []
for country, label in zip(countries, kmeans.labels_):
    dic[f'cluster{label}'].append(country)

for i in range(5):
    print(f'cluster{i}:')
    print(' '.join(dic[f'cluster{i}']))

'''
cluster0:
Peru Cuba Dominica Philippines Guyana Tuvalu Honduras Bahamas Chile Jamaica Venezuela Belize Ecuador Fiji Samoa Suriname Nicaragua
cluster1:
Thailand China Taiwan Japan Qatar Indonesia Iraq Jordan Bahrain Lebanon Afghanistan Syria Bangladesh Libya Greenland Iran Nepal Laos Australia Egypt Oman Bhutan Pakistan Vietnam
cluster2:
Bulgaria Poland Serbia Macedonia Hungary Uzbekistan Estonia Moldova Tajikistan Croatia Albania Slovenia Romania Armenia Belarus Cyprus Turkmenistan Russia Latvia Kazakhstan Kyrgyzstan Greece Lithuania Montenegro Georgia Azerbaijan Turkey Slovakia Ukraine
cluster3:
Ghana Eritrea Guinea Zambia Namibia Liberia Tunisia Zimbabwe Angola Uganda Burundi Gabon Rwanda Algeria Sudan Niger Botswana Senegal Nigeria Mozambique Malawi Somalia Mali Kenya Mauritania Madagascar Gambia
cluster4:
Sweden Uruguay Switzerland Germany Ireland Finland Canada Italy England Belgium Portugal Malta Spain Austria France Morocco Norway Liechtenstein Denmark
'''

with open('./countries_vec.pickle', mode='wb') as f1, open('./countries.pickle', mode='wb') as f2:
    pickle.dump(countries_vec, f1)
    pickle.dump(countries, f2)