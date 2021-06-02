from knock40 import Morph

class Chunk:
    def __init__(self, morphs, dst):
        self.morphs = morphs
        self.phrase = ''.join([morph.surface for morph in morphs])
        self.dst = dst #係り先文節id
        self.srcs = [] #係り元文節id

morphs = [] 
chunks = [] #文(Chunkオブジェクトのリスト)
sentences = [] #文のリスト
with open('ai.ja.txt.parsed') as file:
    for line in file:
        if line == '\n':
            continue

        #形態素をmorphsに追加
        if line != 'EOS\n' and line[0] != '*':
            line = line.replace('\n', '').split('\t')
            morph = [line[0]]
            morph.extend(line[1].split(',')) 
            morphs.append(Morph(morph))

        #EOSか係り受け関係の時、chunksに追加
        elif len(morphs) > 0:
            chunks.append(Chunk(morphs, dst))
            morphs = []
        
        #係り受け関係の時、dst更新
        if line[0] == '*':
            line = line.replace('\n', '').split()
            dst = int(line[2].strip('D'))

        #EOSで文節がある時、sentencesに追加
        if line == 'EOS\n' and len(chunks) > 0:
            for i, chunk in enumerate(chunks):
                if chunk.dst != -1: #係り先が存在
                    chunks[chunk.dst].srcs.append(i) #係り先のsrcsにidを追加
            sentences.append(chunks)
            chunks = []

if __name__ == '__main__':
    for sentence in sentences:
        for chunk in sentence:
            print(f'{chunk.phrase} {chunk.dst} {chunk.srcs}')
