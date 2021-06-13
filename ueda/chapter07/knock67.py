from sklearn.cluster import KMeans
from collections import defaultdict
from knock60 import struct_vector

def country_vector():
    model = struct_vector()
    country = []
    country_name = []
    with open(r'C:\Git\CountryName.txt') as f:
        for line in f:
            country.append(model[line.strip()])
            country_name.append(line.strip())
    return country, country_name

if __name__ == "__main__":
    country, country_name = country_vector()
    kmeans = KMeans(n_clusters=5, random_state=2021).fit_predict(country)
    print(kmeans)
    cluster = defaultdict(list)
    for i in range(len(country_name)):
        cluster[kmeans[i]].append(country_name[i])
    for i in range(5):
        print("Cluster: " + str(i))
        print(cluster[i])
