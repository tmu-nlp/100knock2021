from knock30 import get_nekoList
from collections import defaultdict
from pprint import pprint

d1 = defaultdict(lambda:0)
neko = get_nekoList()
for line in neko:
    for mor in line:
        if mor["pos"] != "記号":
            d1[mor["base"]] += 1

with open("knock35.txt", "w") as f:
    for k,v in sorted(d1.items(),key=lambda x:x[1],reverse=True):
        f.write(f"{k}:{v}\n")