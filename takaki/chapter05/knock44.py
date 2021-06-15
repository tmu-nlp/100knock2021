import pydot
from graphviz import Digraph
from knock40 import Morph
from knock41 import Chunk, Sentence, Document
from typing import List


def chunk2str(chunk: Chunk) -> str:
    return ''.join([
        morph.surface
        for morph in chunk.morphs
        if morph.pos != '記号'
    ])


def pos(chunk: Chunk) -> List[str]:
    return [
        morph.pos
        for morph in chunk.morphs
    ]


# ----------
with open('./ai.ja.txt.parsed', 'r') as f:
    lines = f.readlines()
d = Document.parse(lines)
sentence = d.sentences[1]
# ----------

edges = []

for idx, chunk in enumerate(sentence.chunks):
    if chunk.dst != -1:
        x = f'{str(idx)}: {chunk2str(chunk)}'
        y = f'{str(chunk.dst)}: {chunk2str(sentence.chunks[chunk.dst])}'
        edges.append([x, y])

g = pydot.graph_from_edges(edges, directed=True)
g.write_png('./tmp/knock44.png')
