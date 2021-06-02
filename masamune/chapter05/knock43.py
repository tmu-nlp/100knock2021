from knock41 import sentences

for sentence in sentences:
    for chunk in sentence:
        srcs_phrase = ''
        dst_phrase = ''

        n_flg = False
        for morph in chunk.morphs:
            if morph.pos != '記号':
                srcs_phrase += morph.surface
            if morph.pos == '名詞':
                n_flg = True
                
        v_flg = False
        if chunk.dst != -1 and n_flg:
            for morph in sentence[chunk.dst].morphs:
                if morph.pos != '記号':
                    dst_phrase += morph.surface
                if morph.pos == '動詞':
                    v_flg = True
        
        if v_flg:
            print(f'{srcs_phrase}\t{dst_phrase}')