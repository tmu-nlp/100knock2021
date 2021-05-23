from knock30 import get_nekoList
from collections import defaultdict

d1 = defaultdict(lambda:0)
neko = get_nekoList()
for line in neko:
    for ele in line:
        if ele["pos"] == "動詞":
            d1[ele["surface"]] += 1
print(len(d1))