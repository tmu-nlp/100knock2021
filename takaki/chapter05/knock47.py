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
            for i, src in enumerate(chunk.srcs):
                if len(sentence.chunks[src].morphs) != 2 \
                or sentence.chunks[src].morphs[0].pos1 != 'サ変接続' \
                or sentence.chunks[src].morphs[1].surface != 'を':
                    continue
                a = sentence.chunks[src].morphs[0].surface + \
                    sentence.chunks[src].morphs[1].surface + \
                    morph.base
                b = []
                c = []
                for x in chunk.srcs[:i] + chunk.srcs[i+1:]:
                    d = [
                        m.surface
                        for m in sentence.chunks[x].morphs
                        if m.pos == '助詞'
                    ]
                    if len(d) > 0:
                        b += d
                        c.append(''.join([
                            m.surface
                            for m in sentence.chunks[x].morphs
                        ]))
                if len(b) > 0:
                    b = sorted(list(set(b)))
                    l.append(f"{a}\t{' '.join(b)}\t{' '.join(c)}\n")
with open('tmp/knock47.txt', 'w') as f:
    f.writelines(l)
