from os import write
from knock41 import sentence_set

with open('./ans48.txt','w')as f:
    for sentence in sentence_set:
        for chunk in sentence.chunks:
            if '名詞' in [morph.pos for morph in chunk.morphs]:
                path=[''.join(morph.surface for morph in chunk.morphs if morph.pos!='記号')]
                while chunk.dst!=-1:
                    path.append(''.join(morph.surface for morph in sentence.chunks[chunk.dst].morphs if morph.pos!='記号'))
                    chunk=sentence.chunks[chunk.dst]
                f.write('->'.join(path))
            f.write('\n')