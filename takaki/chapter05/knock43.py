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

for chunk in sentence.chunks:
    if chunk.dst != -1:
        x = chunk2str(chunk)
        y = chunk2str(sentence.chunks[chunk.dst])
        if '名詞' not in pos(chunk):
            continue
        if '動詞' not in pos(sentence.chunks[chunk.dst]):
            continue
        print(f'{x}\t{y}')
