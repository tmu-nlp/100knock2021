from graphviz import Graph
from graphviz import Digraph
from knock41 import sentences
from knock42 import words

dg = Digraph(format='png')

if __name__ == "__main__":
    for i, chunk in enumerate(sentences[2]):
        dg.node(words[i])
    for i, chunk in enumerate(sentences[2]): 
        if chunk.dst != -1:
            dg.edge(words[i], words[chunk.dst])
    dg.render('./dgraph', view=True)