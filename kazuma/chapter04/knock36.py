from knock30 import get_nekoList
from collections import defaultdict
from pprint import pprint
import matplotlib.pyplot as plt

d1 = defaultdict(lambda:0)
neko = get_nekoList()
for line in neko:
    for mor in line:
        if mor["pos"] != "記号":
            d1[mor["base"]] += 1
d1 = sorted(d1.items(),key=lambda x:x[1],reverse=True)[:10]
plt.bar([i[0] for i in d1],[i[1] for i in d1])
plt.show()