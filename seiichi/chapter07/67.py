import gensim
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans

vec = gensim.models.KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True)

# https://gist.github.com/cupnoodlegirl/ba10cf7a412a1840714c
countries = pd.read_csv("./data/country_list.csv")
countries = list(countries["ISO 3166-1に於ける英語名"].array)

country_vecs = []
country_names = []
for country in countries:
    if country in vec:
        country_vecs.append(vec[country])
        country_names.append(country)
        
kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(country_vecs)
for i in range(5):
    cluster = np.where(kmeans.labels_ == i)[0]
    print("cluster:", i)
    print(", ".join([country_names[j] for j in cluster]))

"""
cluster: 0
Algeria, Angola, Uganda, Ethiopia, Eritrea, Ghana, Gabon, Cameroon, Gambia, Guinea, Kenya, Comoros, Congo, Zambia, Djibouti, Zimbabwe, Sudan, Swaziland, Senegal, Somalia, Chad, Tunisia, Togo, Nigeria, Namibia, Niger, Burundi, Benin, Botswana, Madagascar, Malawi, Mali, Mauritania, Mozambique, Libya, Liberia, Rwanda, Lesotho
cluster: 1
Argentina, Aruba, Uruguay, Ecuador, Canada, Cuba, Guatemala, Colombia, Spain, Suriname, Chile, Nicaragua, Haiti, Panama, Paraguay, Brazil, Peru, Portugal, Honduras, Mexico
cluster: 2
Iceland, Ireland, Azerbaijan, Albania, Armenia, Andorra, Italy, Ukraine, Estonia, Austria, Netherlands, Kazakhstan, Cyprus, Greece, Georgia, Croatia, Switzerland, Sweden, Slovakia, Slovenia, Serbia, Denmark, Germany, Turkey, Norway, Hungary, Finland, France, Bulgaria, Belarus, Belgium, Poland, Malta, Monaco, Montenegro, Latvia, Lithuania, Liechtenstein, Romania, Luxembourg
cluster: 3
Anguilla, Guernsey, Guyana, Curaçao, Kiribati, Guadeloupe, Guam, Greenland, Grenada, Samoa, Gibraltar, Jersey, Jamaica, Seychelles, Tuvalu, Tokelau, Dominica, Tonga, Nauru, Antarctica, Niue, Vanuatu, Bahamas, Bermuda, Palau, Barbados, Pitcairn, Fiji, Belize, Mayotte, Martinique, Mauritius, Maldives, Montserrat, Réunion
cluster: 4
Afghanistan, Yemen, Israel, Iraq, India, Indonesia, Uzbekistan, Egypt, Australia, Oman, Qatar, Cambodia, Kyrgyzstan, Kuwait, Singapore, Thailand, Tajikistan, China, Turkmenistan, Japan, Nepal, Bahrain, Pakistan, Bangladesh, Philippines, Bhutan, Macao, Malaysia, Myanmar, Morocco, Mongolia, Jordan, Lebanon

"""