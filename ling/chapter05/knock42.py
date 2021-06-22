
from knock41 import sentence_set
sent=sentence_set[2]
print(sentence_set)
for chunk in sent.chunks:
    if chunk.dst !=-1:
        modifier=''.join([morph.surface if morph.pos != '記号' else '' for morph in chunk.morphs])
        modifiee=''.join([morph.surface if morph.pos != '記号' else '' for morph in sent.chunks[chunk.dst].morphs])
        print(modifier+' '+modifiee)

