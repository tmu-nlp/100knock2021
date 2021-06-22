from sklearn.cluster import KMeans
from knock60 import load_vectors
from collections import defaultdict
from pprint import pprint
def get_nation_names():
    with open("data/nation_names", "r") as f:
        names = []
        for line in f:
            names.append(line.split("\t")[1].replace(" ","_"))
    return names

def get_nation_vectors():
    word_vectors = load_vectors()
    nation_names = get_nation_names()
    nation_vectors = []
    keyErr_cnt = 0
    with open("data/not_exist_nations.txt","w")as f:
        for name in nation_names:
            try:
                nation_vectors.append(word_vectors[name])
            except(KeyError):
                keyErr_cnt += 1
                f.write(f"{name}\n")
    print("KeyError:", keyErr_cnt)
    return [[i,j] for i,j in zip(nation_names, nation_vectors)]
        
def knock67():
    nation_name_and_vec = get_nation_vectors()
    pred = KMeans(n_clusters=5).fit_predict([nation[1] for nation in nation_name_and_vec])
    de_dict1 = defaultdict(lambda:[])
    for i in range(len(nation_name_and_vec)):
        de_dict1[pred[i]].append(nation_name_and_vec[i][0])
    for key, value in sorted(de_dict1.items(), key = lambda x :x[0]):
        print(key, value)

if __name__ == "__main__":
    knock67()