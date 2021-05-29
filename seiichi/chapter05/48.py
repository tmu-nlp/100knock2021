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

def passMaker(sentence, dst):
    kakari_saki = sentence.sentence[int(dst)]
    next_kakari = sentence.dst[int(dst)]
    
    return kakari_saki, next_kakari
    

def complete_list(sentence):
    com_list = []
    
    for index, dst in enumerate(sentence.dst):
        com_list_temp = []
        if dst != "-1":
            com_list_temp.append(sentence.sentence[index])
            next_kakari = dst
            while next_kakari != "-1":
                kakari_saki, next_kakari = passMaker(sentence, next_kakari)
                com_list_temp.append(kakari_saki)
        com_list.append(com_list_temp)
    
    return com_list

def out(sentence):
    for i in complete_list(sentence):
        print(re.sub(r"。|、", "", " -> ".join(i))) 

out(all_sent[5])
out(all_sent[8])
