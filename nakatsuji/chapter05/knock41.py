'''
40に加えて，文節を表すクラスChunkを実装せよ．
このクラスは
形態素(Morphオブジェクト）のリスト（morphs），
係り先文節インデックス番号（dst），
係り元文節インデックス番号のリスト（srcs）
をメンバ変数に持つこととする．
さらに，入力テキストの係り受け解析結果を読み込み，
１文をChunkオブジェクトのリストとして表現し，
冒頭の説明文の文節の文字列と係り先を表示せよ．
本章の残りの問題では，ここで作ったプログラムを活用せよ．
'''
'''
1. *
2. 文節番号
3. 係り先の文節番号(係り先なし:-1)
4. 主辞の形態素番号/機能語の形態素番号
5. 係り関係のスコア(大きい方が係りやすい)
'''

from knock40 import Morph
class Chunk:
    def __init__(self):
        self.morphs = []
        self.index = -1
        self.dst = -1
        self.srcs = []
    
    def __str__(self):
        surfaces = [morph.surface for morph in self.morphs]
        return \
            f'表層形：{surfaces} 係り先：{self.dst} 係り元：{self.srcs}'


with open('ai.ja.txt.parsed') as f:
    lines = f.readlines()
    sentences = [] #sen のリスト
    sen = [] # chunk のリスト
    chunk = Chunk()
    for i, line in enumerate(lines):
        line = line.strip()
        if line[0] == '*':
            line = line.split()
            if line[1] != '0':
                sen.append(chunk)
            chunk = Chunk()
            chunk.index = int(line[1])
            chunk.dst = int(line[2][:-1])
        elif line == 'EOS':
            if lines[i-1] == "EOS":
                continue
            elif len(chunk.morphs) > 0:
                sen.append(chunk)
                sentences.append(sen)
                sen = []
            chunk = Chunk()
        else:
            surface, info = line.split('\t')
            info = info.split(',')
            base = info[-3]
            pos = info[0]
            pos1 = info[1]
            morph = Morph(surface, base, pos, pos1)
            chunk.morphs.append(morph)
    #srcs
    for sen in sentences:
        for chunk in sen:
            dst = chunk.dst
            if not dst == -1:
                sen[dst].srcs.append(chunk.index)
if __name__ == "__main__":
    #out
    with open('chunks.txt', 'w') as f:
        for sen in sentences:
            print('------------',file=f)
            for chunk in sen:
                print(chunk, file=f)
