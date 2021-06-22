import pydot
from knock41 import sentence_set
from graphviz import Digraph

sentence = sentence_set[2]
g=Digraph(format='png')
for id, chunk in enumerate(sentence.chunks):
  if int(chunk.dst) != -1:
    modifier = ''.join([morph.surface if morph.pos != '記号' else '' for morph in chunk.morphs] + ['(' + str(id) + ')'])
    g.node(modifier)
    modifiee = ''.join([morph.surface if morph.pos != '記号' else '' for morph in sentence.chunks[int(chunk.dst)].morphs] + ['(' + str(chunk.dst) + ')'])
    g.node(modifiee)
    g.edge(modifier, modifiee)
#g.view()
g.render('./graph', view=True)