from knock40 import Morph

class Chunk:
    def __init__(self):
        self.morphs= []
        self.dst = -1
        self.srcs = []
    
    def __str__(self):
        return 'surface:{}\tdst:{}\tsrcs:{}'.format(''.join(list(a.surface for a in self.morphs)), self.dst, self.srcs)

def load_chunk():
    with open(r'/Users/Naoya/Downloads/ai.ja.txt.parsed', encoding="utf-8") as f:
        chunks = {}
        srcs_dicts = {}
        for line in f:
            col = line.split('\t')
            if col[0]=='EOS\n' and len(chunks) != 0:
                for key in srcs_dicts.keys():
                    chunks[key].srcs = srcs_dicts[key]
                yield chunks
                chunks = {}
                srcs_dicts = {}
            elif line[0]=='*':
                s_line = line.split(' ')
                dst = s_line[2].rstrip('D')
                id = s_line[1]
                if id not in chunks:
                    chunks[id] = Chunk()
                chunks[id].dst = dst
                if dst != str('-1'):
                    if dst not in srcs_dicts:
                        srcs_dicts[dst] = []
                    srcs_dicts[dst].append(id)
            elif col[0]!='EOS\n' and line[0]!='*':
                other = col[1].split(',')
                chunks[id].morphs.append(Morph(col[0], other[6], other[0], other[1]))

if __name__ == "__main__":
    for line in load_chunk():
        lchunks = list(line.values())
        for l in lchunks:
            print(l)