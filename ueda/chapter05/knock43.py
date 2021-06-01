from knock41 import load_chunk

noun_phrase = ''
verb_phrase = ''
noun_b, verb_b = False, False
for line in load_chunk():
    lchunks = list(line.values())
    for l in lchunks:
        if int(l.dst) != -1:
            for word in l.morphs:
                if word.pos != '記号':
                    noun_phrase+=word.surface
                if word.pos == '名詞':
                    noun_b = True
            for word in line[l.dst].morphs:
                if noun_b != True: break
                if word.pos != '記号':
                    verb_phrase+=word.surface
                if word.pos == '動詞':
                    verb_b = True
            if(noun_b == 1 and verb_b==1):
                print(noun_phrase + '\t' + verb_phrase)
            noun_phrase = ''
            verb_phrase = ''
            noun_b, verb_b = False, False