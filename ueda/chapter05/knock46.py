from knock41 import load_chunk

with open('/Users/Naoya/Downloads/case.txt', 'w', encoding="utf-8") as f:
    for line in load_chunk():
        srcs = []
        base = []
        lchunks = list(line.values())
        for l in lchunks:
            for word in l.morphs:
                if word.pos == '動詞':
                    srcs.append(l.srcs)
                    base.append(word.base)
        if(len(base)>0):
            for src in srcs[0]:
                case = []
                phrase = []
                for tmp in line[src].morphs:
                    if tmp.pos == '助詞':
                        case.append(tmp.surface)
                        phrase.append(''.join(list(a.surface for a in line[src].morphs if a.pos != '記号')))
                        if(len(case)>0):
                            f.write(base[0]+"\t"+' '.join(case)+"\t"+' '.join(phrase)+"\n")
