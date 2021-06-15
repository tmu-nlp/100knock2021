from knock40 import Morph
from knock41 import Chunk, Sentence, Document


# ----------
with open('./ai.ja.txt.parsed', 'r') as f:
    lines = f.readlines()
d = Document.parse(lines)
# sentence = d.sentences[1]
# ----------

l = []
for sentence in d.sentences:
    for chunk in sentence.chunks:
        for morph in chunk.morphs:
            if morph.pos != '動詞':
                break
            x, y = [], []
            for src in chunk.srcs:
                x += [
                    m.surface
                    for m in sentence.chunks[src].morphs
                    if m.pos == '助詞'
                ]
                y += ''.join([
                    m.surface
                    for m in sentence.chunks[src].morphs
                ])
            if len(x) > 0:
                l.append(f"{morph.base}\t{' '.join(sorted(x))}\t{' '.join(y)}\n")
with open('tmp/knock46.txt', 'w') as f:
    f.writelines(l)
