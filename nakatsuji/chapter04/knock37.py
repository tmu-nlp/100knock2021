import matplotlib.pyplot as plt
from knock30 import sentences
from collections import defaultdict

co_neko = defaultdict(lambda: 0)
for sen in sentences:
    if '猫'in [word['surface'] for word in sen]:
        for word in sen:
            if word['pos'] != '記号' :
                co_neko[word['base']] += 1

del co_neko['猫']
co_neko = sorted(co_neko.items(), key=lambda x: x[1], reverse=True)


if __name__ == '__main__':

    left, height = [], []
    for w in co_neko[0:10]:
        left.append(w[0])
        height.append(w[1])
    
    plt.figure()
    plt.bar(left, height)
    plt.show()