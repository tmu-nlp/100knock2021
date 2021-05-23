from knock30 import get_nekoList
from collections import defaultdict
from pprint import pprint

d1 = defaultdict(lambda:0)
err_l = []
neko = get_nekoList()
for line in neko:
    list1 = []
    for i,mor in enumerate(line):
        if mor["surface"]=="の" and mor["pos"]=="助詞":
            list1.append(i)
    if list1:
        for i in list1:
            try:
                if line[i-1]["pos"]=="名詞" and line[i+1]["pos"]=="名詞":
                    d1[line[i-1]["surface"]+line[i]["surface"]+line[i+1]["surface"]] += 1
            except IndexError :
                err_l.append(line)
print(d1)
pprint(err_l)