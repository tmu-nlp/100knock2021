from knock41 import sentence_set
sentence = sentence_set[2]
for chunk in sentence.chunks:
  if chunk.dst != -1:
    modifier = ''.join([morph.surface if morph.pos != '記号' else '' for morph in chunk.morphs])
    modifier_pos = [morph.pos for morph in chunk.morphs]
    modifiee = ''.join([morph.surface if morph.pos != '記号' else '' for morph in sentence.chunks[int(chunk.dst)].morphs])
    modifiee_pos = [morph.pos for morph in sentence.chunks[chunk.dst].morphs]
    if '名詞' in modifier_pos and '動詞' in modifiee_pos:
      print(modifier+' '+modifiee)