from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
import numpy as np

f = open('ans-knock64.txt', 'r')

#国名のセットをアナロジーのファイルから取得
countries = set()
for line in f:
    l = line.strip().split(' ')

    if l[2] in countries or l[1] in countries:
        continue
    elif 'capital-common-countries' in l[0] or 'capital-world' in l[0]:
        countries.add(l[2])
    elif 'currency' in l[0] or  'gram6-nationality-adjective' in l[0]:
        countries.add(l[1])

#国名の単語ベクトルを取得
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
countries = list(countries)
countries_vec = [model[country] for country in countries]

# k-meansクラスタリング
kmeans = KMeans(n_clusters=5)
kmeans.fit(countries_vec)
for i in range(5):
    cluster = np.where(kmeans.labels_ == i)[0]
    print('cluster', i)
    print(', '.join([countries[k] for k in cluster]))

'''

cluster 0
Russia, Albania, Croatia, Slovakia, Lithuania, Ukraine, Azerbaijan, Tajikistan, Georgia, Macedonia, Uzbekistan, Slovenia, Bulgaria, Kazakhstan, Kyrgyzstan, Estonia, Armenia, Montenegro, Belarus, Cyprus, Latvia, Turkey, Hungary, Romania, Turkmenistan, Moldova, Poland, Serbia
cluster 1
Lebanon, Iran, Jordan, Malta, Norway, Austria, Liechtenstein, Germany, Iraq, Syria, Egypt, Portugal, Afghanistan, Belgium, Denmark, Switzerland, France, Qatar, Canada, Sweden, Libya, Algeria, USA, Tunisia, Greenland, Morocco, Spain, Finland, England, Europe, Greece, Italy, Ireland
cluster 2
Fiji, Japan, Oman, Philippines, India, Bangladesh, Taiwan, Laos, China, Samoa, Vietnam, Tuvalu, Australia, Bahrain, Korea, Bhutan, Pakistan, Cambodia, Thailand, Nepal, Malaysia, Indonesia
cluster 3
Namibia, Kenya, Zimbabwe, Burundi, Sudan, Somalia, Angola, Nigeria, Uganda, Mozambique, Botswana, Malawi, Rwanda, Liberia, Niger, Eritrea, Zambia, Mali, Mauritania, Ghana, Madagascar, Gabon, Gambia, Guinea, Senegal
cluster 4
Cuba, Brazil, Ecuador, Nicaragua, Suriname, Venezuela, Peru, Honduras, Mexico, Guyana, Bahamas, Uruguay, Chile, Belize, Jamaica, Dominica, Argentina

'''