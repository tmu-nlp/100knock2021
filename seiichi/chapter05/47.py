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

def g(sentence, output="default"):
    saki_index = [] ; saki_list = [] ; saki_nextV_list = []
    for index, morphs in enumerate(sentence.morphs):
        for index2, morph in enumerate(morphs):
            flag = False
            if morph.pos1 == "サ変接続":
                try:
                    if morphs[index2 + 1].surface == "を" and sentence.morphs[index + 1 ][0].pos == "動詞":
                        flag = True
                except IndexError:
                    pass
                
                if flag:
                    saki_index.append(index)
                    saki_list.append(morph.base)
                    saki_nextV_list.append(sentence.morphs[index + 1 ][0].base)
                    break
    
    moto_list = [] ; moto_list_temp = [] ; moto_phrase_list = [] ; moto_phrase_temp = []
    for saki in saki_index:
        for moto in sentence.srcs[saki]:
            if moto == "-1":
                next
            else:
                for index_moto, morph in enumerate(sentence.morphs[int(moto)]):
                    if morph.pos == "助詞":
                        if len(re.findall(sentence.sentence[int(moto)], sentence.sentence[int(saki)])) == 0:
                            moto_phrase_temp.append(sentence.sentence[int(moto)])
                            moto_list_temp.append(morph.surface)

        for moto in sentence.srcs[saki+1]:
            if moto == "-1":
                next
            else:
                for index_moto, morph in enumerate(sentence.morphs[int(moto)]):
                    if morph.pos == "助詞":
                        if len(re.findall(sentence.sentence[int(moto)], sentence.sentence[int(saki)])) == 0:
                            moto_phrase_temp.append(sentence.sentence[int(moto)])
                            moto_list_temp.append(morph.surface)
       
        
        moto_list.append(moto_list_temp) ; moto_list_temp = [] 
        moto_phrase_list.append(moto_phrase_temp) ; moto_phrase_temp = []

    output_list = list(zip(saki_list, saki_nextV_list, moto_list, moto_phrase_list))

    for a, b, c, d in output_list:
        if a == [] or b == [] or c == [] or d == []:
            next
        else:
            if output == "default":
                return (a + "を" + b + "\t" + re.sub(r"。|、", "", " ".join(c)) +  "\t" +  re.sub(r"。|、", "", " ".join(d)))
            else:
                print("Output style not defined.")

kumiawase_list = []
for sentence in all_sent:
    if isinstance(g(sentence, output="default"), str):
        kumiawase_list.append(g(sentence, output="default").split("\t")[0])

counter = Counter(kumiawase_list) ; count = 0
for word, cnt in counter.most_common():
    count += 1
    print (word, cnt)
    if count == 10:
        break

kumiawase_list = []
for sentence in all_sent:
    if isinstance(g(sentence, output="default"), str):
        moto = g(sentence, output="default").split("\t")[0]
        saki = (g(all_sent[868], output="default").split("\t"))[1].split(" ")

        if len(saki) == 1:
            kumiawase_list.append(str(moto + " " + saki[0]))
        else:
            for i in saki:
                kumiawase_list.append(str(moto + " " + i))

cnt = Counter(kumiawase_list)
for k, v in cnt.most_common(10):
    print(k, v)
