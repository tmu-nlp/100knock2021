class Chunk:
    def __init__(self,morphs,dst):
        self.morphs=morphs
        self.dst=dst
        self.srcs=[]

class Sentence:
    def __init__(self,chunks):
        self.chunks=chunks
        for i,chunk in enumerate(self.chunks):
            if chunk.dst!= None:
                self.chunks[chunk.dst].srcs.append(i)
class Morph:
    #インスタンスの初期化
    def __init__(self,morph):
        surface,attr=morph.split('\t')
        attr=attr.split(',')
        self.surface=surface
        self.base=attr[6]
        self.pos=attr[0]
        self.pos1=attr[1]

file_name="./ai.ja.txt.parse"

sentence_set=[]
morphs=[]
chunks=[]
with open(file_name,'r') as f:
    for line in f:
        if line[0]=='*':# * の行は係受け関係を表す行
            if len(morphs)>0:#初めて読んだ*の次の内容がchunkに保存
                chunks.append(Chunk(morphs,dst))
                morphs=[]
            dst=int(line.split(' ')[2].rstrip('D'))
        elif line !='EOS\n':
            morphs.append(Morph(line))
        else:
            chunks.append(Chunk(morphs,dst))
            sentence_set.append(Sentence(chunks))
            morphs=[]
            chunks=[]
            dst=None
'''
for c in sentence[2].chunks:
    print([morph.surface for morph in c.morphs],c.dst,c.srcs)
'''