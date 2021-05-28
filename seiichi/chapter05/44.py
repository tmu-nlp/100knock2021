import re
import CaboCha
import pickle
import pydot
from graphviz import Digraph
from common import Morph, Chunk

with open("data/all_sent.pkl", "rb") as f:
    all_sent = pickle.load(f)

def kakari_graphic(sentence):
    kakari_g = Digraph(comment='kakariuke', format='pdf')
    for index, phrase in enumerate(sentence.sentence):
        phrase = re.sub(r"。|、", "", phrase)
        kakari_g.node(str(index), phrase)
        
    for index, kakari_saki in enumerate(sentence.dst):
        if kakari_saki != "-1":
            kakari_g.edge(str(index), kakari_saki)
    
    return kakari_g

kakari_graphic(all_sent[9]).render('data/kakari')
