from knock41 import load_chunk

with open('c:\Git\case_pattern.txt', 'w', encoding="utf-8") as f:
    for line in load_chunk():
        srcs = []
        base = []
        lchunks = list(line.values())
        for l in lchunks:
            if int(l.dst) != -1:
                for word in l.morphs:
                    if word.pos == '動詞':
                        srcs.append(l.srcs)
                        base.append(word.base)
        if(len(base)>0):
            for src in srcs[0]:
                case = []
                phrase = []
                if len(line[src].morphs) == 2 and line[src].morphs[0].pos == '名詞' and line[src].morphs[0].pos1 == 'サ変接続' and line[src].morphs[1].surface == 'を':
                    base[0] = line[src].morphs[0].surface + line[src].morphs[1].surface + base[0]
                    print(base[0])
                    for src2 in srcs[0]:
                        if src2 == src:
                            continue
                        for a in line[src2].morphs:
                            if a.pos == '助詞':
                                case.append(a.surface)
                                phrase.append(''.join(list(b.surface for b in line[src2].morphs if b.pos != '記号')))
                    f.write(base[0]+"\t"+' '.join(case)+"\t"+' '.join(phrase)+"\n")
