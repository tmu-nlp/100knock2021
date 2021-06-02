from graphviz import Digraph
from knock41 import load_chunk

digraph = Digraph(format='png')
for line in load_chunk():
    if line.keys() != 2:
        continue
    for l in line:
        if int(line[l].dst) != -1:
            digraph.edge(''.join(list(a.surface for a in line[l].morphs if a.pos != '記号')), ''.join(list(a.surface for a in line[line[l].dst].morphs if a.pos != '記号')))
    digraph.render('/Users/Naoya/100knock2021/ueda/chapter05/tree')
    