from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from knock60 import load_vectors

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
        
def knock68():
    nation_name_and_vec = get_nation_vectors()
    plt.figure(figsize=(15, 5))
    Z = linkage([nation[1] for nation in nation_name_and_vec], method='ward')
    dendrogram(Z, labels=[nation[0] for nation in nation_name_and_vec])
    plt.show()
if __name__ == "__main__":
    knock68()
