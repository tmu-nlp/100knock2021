import re
import CaboCha
import pickle
import pydot
from collections import Counter
from graphviz import Digraph
import pandas as pd
from common import Morph, Chunk

with open("data/all_sent.pkl", "rb") as f:
    all_sent = pickle.load(f)

def f(sentence):
    saki_index = [] ; saki_list = []
    for index, morphs in enumerate(sentence.morphs):
        for morph in morphs:
            if morph.pos == "動詞":
                saki_index.append(index)
                saki_list.append(morph.base)
                break
    moto_list = [] ; moto_list_temp = []
    for saki in saki_index:
        for moto in sentence.srcs[saki]:
            if moto == "-1":
                next
            else:
                for index_moto, morph in enumerate(sentence.morphs[int(moto)]):
                    if morph.pos == "助詞":
                        moto_list_temp.append(morph.surface)
        moto_list.append(moto_list_temp) ; moto_list_temp = []
    output_list = list(zip(saki_list, moto_list))
    return output_list

c = []
for sentence in all_sent:
    for item in f(sentence):
        moto, saki = item
        if len(saki) == 1:
            c.append(str(moto + " " + saki[0]))
        else:
            for i in saki:
                c.append(str(moto + " " + i))

counter = Counter(c)
count = 0
for word, cnt in counter.most_common():
    count += 1
    print(word, cnt)
    if count == 10:
        break

def pattern(all_sent ,verb):
    l = []
    for sentence in all_sent:
        for item in f(sentence):
            moto, saki = item
            if moto == verb:
                if len(saki) == 1:
                    l.append(str(moto + " " + saki[0]))
                else:
                    for i in saki:
                        l.append(str(moto + " " + i))
    return Counter(l)

for tar in ["する", "見る", "与える"]:
    cnt = pattern(all_sent, verb=tar)
    i = 0
    for k, v in cnt.most_common():
        if i > 10: break
        print(k, v)
        i += 1
