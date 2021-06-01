from graphviz import Digraph
from knock41 import load_chunk

digraph = Digraph(format='png')
for line in load_chunk():
    lchunks = list(line.values())
    for l in lchunks:
        if int(l.dst) != -1:
            diagram.edge(''.join(list(a.surface for a in l.morphs if a.pos != '記号')), ''.join(list(a.surface for a in line[l.dst].morphs if a.pos != '記号')))
digraph.render('tree')