#与えられた文の係り受け木を有向グラフとして可視化せよ．可視化には，Graphviz等を用いるとよい．

import numpy as np
import pydot
from IPython.display import Image, display
from graphviz import Digraph

from knock40 import Morph
from knock41 import Chunks, chunk_list

if __name__ == '__main__':
    f = open('ai.ja.txt.parsed', 'r')
    
    sentences = chunk_list(f)
    edges =[]

    for i, chunk in enumerate(sentences[2]):
        souce = chunk
        destination = sentences[2][chunk.dst]
        if souce.dst != -1:
            s = souce.connect() + '(' + str(i) + ')'
            d = destination.connect() + '(' + str(chunk.dst) + ')'
            edge = (s,d,)
            edges.append(edge)
    
    f.close()

    n = pydot.Node('node')
    n.fontname = 'Hiragino sans'
    g = pydot.graph_from_edges(edges, directed=True)
    g.add_node(n)
    g.write_png('./ans44.png')