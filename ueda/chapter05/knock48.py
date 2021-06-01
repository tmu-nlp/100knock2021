from knock41 import load_chunk

for line in load_chunk():
    lchunks = list(line.values())
    for l in lchunks:
        poslist = [tmp.pos for tmp in l.morphs]
        if '名詞' in poslist and int(l.dst) != -1:
            paths = []
            paths.append(''.join(list(a.surface for a in l.morphs if a.pos != '記号')))
            dst=l.dst
            while dst != '-1':
                paths.append(''.join(list(a.surface for a in line[dst].morphs if a.pos != '記号')))
                dst = line[dst].dst
            print(' -> '.join(paths))
