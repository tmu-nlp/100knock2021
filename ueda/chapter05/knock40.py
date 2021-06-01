class Morph:
    def __init__(self, surface, base, pos, pos1):
        self.surface = surface
        self.base = base
        self.pos = pos
        self.pos1 = pos1
    
    def __str__(self):
        return 'surface:{}\tbase:{}\tpos:{}\tpos1:{}'.format(self.surface, self.base, self.pos, self.pos1)

def load_parsed():
     with open(r'c:\Git\ai.ja.txt.parsed', encoding="utf-8") as f:
         morphs = []
         for line in f:
            col = line.split('\t')
            if col[0]=='EOS\n' and len(morphs) != 0:
                yield morphs
                morphs = []
            elif col[0]!='EOS\n' and line[0]!='*':
                other = col[1].split(',')
                morphs.append(Morph(col[0], other[6], other[0], other[1]))


if __name__ == "__main__":
    for line in load_parsed():
        for word in line:
            print(word)