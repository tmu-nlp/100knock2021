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
l1 = sorted(d1.items(),key=lambda x:x[1],reverse=True)
# plt.hist([i[1] for i in l1],range = [0,20])
plt.hist([i[1] for i in l1])
plt.show()