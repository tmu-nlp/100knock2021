'''
-- knock44 --
与えられた文の係り受け木を有向グラフとして可視化せよ．
可視化には，Graphviz等を用いるとよい．
'''
from knock41 import get_chunk_sentences
from graphviz import Digraph

sentences = get_chunk_sentences()
dg = Digraph()

sentence = sentences[2]
set1 = set()
set2 = set()
for chunk in sentence:
    if chunk.dst == "-1":
        continue
    str1 = "".join([a_mor.surface for a_mor in chunk.morphs if a_mor.pos != "記号"])
    str2 = "".join([a_mor.surface for a_mor in sentence[int(chunk.dst)].morphs if a_mor.pos != "記号"])
    set1.add(str1)
    set1.add(str2)
    set2.add((str1,str2))
for i in set1:
    dg.node(i)
for i in set2:
    dg.edge(i[0],i[1])
dg.render('knock44.gv', view=True) 

        


