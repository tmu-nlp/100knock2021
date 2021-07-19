""'''
44. 与えられた文の係り受け木を有向グラフとして可視化せよ．
可視化には，Graphviz等を用いるとよい．
'''
from knock41 import load_chunk
import pydot
from IPython.display import Image, display_png
from graphviz import Digraph


if __name__ ==  '__main__':
    filepath = './data/ai.ja/ai.ja.txt.parsed'
    res = load_chunk(filepath)
    res = res[7]
    edges = []
    for id, chunk in enumerate(res.chunks):
        if int(chunk.dst) != -1:
            modifier = ''.join([morph.surface if morph.pos != '記号' else '' for morph in chunk.morphs] + ['(' + str(id) + ')'])
            modifiee = ''.join([morph.surface if morph.pos != '記号' else '' for morph in res.chunks[int(chunk.dst)].morphs] + ['(' + str(id) + ')'])
            edges.append([modifier, modifiee])


    n = pydot.Node('node')
    n.fontname = 'IPAGothic'
    g = pydot.graph_from_edges(edges, directed=True)
    g.add_node(n)
    g.write('./ans44.png')
    display_png(Image('./ans44.png'))






