from knock41 import sentences

for sentence in sentences:
    for chunk in sentence:
        srcs_phrase = ''
        dst_phrase = ''

        for morph in chunk.morphs:
            if morph.pos != '記号':
                srcs_phrase += morph.surface
                
        if chunk.dst != -1:
            for morph in sentence[chunk.dst].morphs:
                if morph.pos != '記号':
                    dst_phrase += morph.surface

        print(f'{srcs_phrase}\t{dst_phrase}')