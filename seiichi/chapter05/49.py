import re
import CaboCha
import pickle
import pydot
import itertools
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

def out(sentence):
    kumiawase_list = []
    for i in range(len(sentence.dst)):
        for item in sentence.morphs[i]:
            if item.pos == "名詞":
                kumiawase_list.append(i)
    kumiawase_list = list(itertools.combinations(kumiawase_list, 2))
    
    for i, j in kumiawase_list:
        com_list_temp = [] ; pass_list_i = []
        if sentence.dst[i] != "-1":
            next_kakari = sentence.dst[i]
            while next_kakari != "-1":
                com_list_temp.append(int(next_kakari))
                kakari_saki, next_kakari = passMaker(sentence, next_kakari)        
        pass_list_i = com_list_temp     
        
        com_list_temp = [] ; pass_list_j = []
        if sentence.dst[j] != "-1":
            next_kakari = sentence.dst[j]
            while next_kakari != "-1":
                com_list_temp.append(int(next_kakari))
                kakari_saki, next_kakari = passMaker(sentence, next_kakari)        
        pass_list_j = com_list_temp        
        
        common_pass = list(set(pass_list_i).intersection(set(pass_list_j))) 

        X = re.sub(r"。|、", "", re.sub(sentence.morphs[i][0].surface, "X", sentence.sentence[i]))
        Y = re.sub(r"。|、", "", re.sub(sentence.morphs[j][0].surface, "Y", sentence.sentence[j]))
        
        output = []
        if j in pass_list_i:
            output.append(X)
            for index in pass_list_i:
                if j == index: 
                    break
                else:
                    output.append(sentence.sentence[index])
            output.append(Y)
            
            print(" -> ".join(output))
            
        else:
            k = min(common_pass)
            
            com_list_temp = [] ; pass_list_iTOk = []
            com_list_temp.append(X)
            next_kakari = sentence.dst[i]
            while next_kakari != "-1" and next_kakari != k:
                if sentence.sentence[int(next_kakari)] != sentence.sentence[int(k)]:
                    com_list_temp.append(re.sub(r"。|、", "", sentence.sentence[int(next_kakari)]))
                kakari_saki, next_kakari = passMaker(sentence, next_kakari)        
            pass_list_iTOk = com_list_temp 

            com_list_temp = [] ; pass_list_jTOk = []
            com_list_temp.append(Y)
            next_kakari = sentence.dst[j]
            while next_kakari != "-1" and next_kakari != k:
                if sentence.sentence[int(next_kakari)] != sentence.sentence[int(k)]:
                    com_list_temp.append(re.sub(r"。|、", "", sentence.sentence[int(next_kakari)]) )
                kakari_saki, next_kakari = passMaker(sentence, next_kakari)        
            pass_list_jTOk = com_list_temp                 

            print( " -> ".join(pass_list_iTOk), "|", " -> ".join(pass_list_jTOk), "|", re.sub(r"。|、", "", sentence.sentence[int(k)]))

out(all_sent[5])
out(all_sent[7])
