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
    saki_index = [] ; saki_list = []
    for index, morphs in enumerate(sentence.morphs):
        for morph in morphs:
            if morph.pos == "動詞":
                saki_index.append(index)
                saki_list.append(morph.base)
                break
    
    moto_list = [] ; moto_list_temp = [] ; moto_phrase_list = [] ; moto_phrase_temp = []
    for saki in saki_index:
        for moto in sentence.srcs[saki]:
            if moto == "-1":
                next
            else:
                for index_moto, morph in enumerate(sentence.morphs[int(moto)]):
                    if morph.pos == "助詞":
                        moto_phrase_temp.append(sentence.sentence[int(moto)])
                        moto_list_temp.append(morph.surface)
        
        moto_list.append(moto_list_temp) ; moto_list_temp = [] 
        moto_phrase_list.append(moto_phrase_temp) ; moto_phrase_temp = []

    output_list = list(zip(saki_list, moto_list, moto_phrase_list))

    for a, b, c in output_list:
        if a == [] or b == [] or c == []:
            next
        print(a + "\t" +  " ".join(b) +  "\t" +  " ".join(c))

for i in range(6):
    g(all_sent[i], output="default")
