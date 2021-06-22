from sklearn.cluster import KMeans
from collections import *
df = pd.read_csv('country_list.csv')
countries = list(df.iloc[:, 1])

if __name__ == "__main__":
    country_vec = []
    country_name = []
    for country in countries:
        if country in model.vocab:
            country_vec.append(model[country])
            country_name.append(country)
    
    kmeans = KMeans(n_clusters=5)
    y = kmeans.fit_predict(country_vec)

    clusters = defaultdict(list)
    for i, country in zip(y, country_name):
        clusters[i].append(country)
    clusters = dict(sorted(clusters.items(), key=lambda x:x[0]))
    for n, countries in clusters.items():
        print(f'Cluster{n} : {countries}')