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
        if '名詞' not in [m.pos for m in chunk.morphs]:
            continue
        path = []
        path.append(''.join([
            m.surface
            for m in chunk.morphs
        ]))
        next_chunk = chunk
        while next_chunk.dst != -1:
            path.append(''.join([
               m.surface
                for m in sentence.chunks[next_chunk.dst].morphs
            ]))
            next_chunk = sentence.chunks[next_chunk.dst]
        l.append(' -> '.join(path) + '\n')
with open('tmp/knock48.txt', 'w') as f:
    f.writelines(l)
