from knock41 import sentences
from graphviz import Digraph

sentence = sentences[1]
graph = Digraph(format='png')
for i, chunk in enumerate(sentence):
    if chunk.dst == -1:
        continue #ネストを減らせる(early continue)
    #始点
    s_phrase = [f'{i} '] + [morph.surface for morph in chunk.morphs if morph.pos != '記号']
    #終点
    e_phrase = [f'{chunk.dst} '] + [morph.surface for morph in sentence[chunk.dst].morphs if morph.pos!= '記号']
    graph.edge(''.join(s_phrase), ''.join(e_phrase))

graph.render("image/knock44")