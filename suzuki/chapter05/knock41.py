'''

40に加えて，文節を表すクラスChunkを実装せよ．このクラスは
形態素（Morphオブジェクト）のリスト（morphs），係り先文節インデックス番号（dst），係り元文節インデックス番号のリスト（srcs）
をメンバ変数に持つこととする．さらに，入力テキストの係り受け解析結果を読み込み，１文をChunkオブジェクトのリストとして表現し，
冒頭の説明文の文節の文字列と係り先を表示せよ．本章の残りの問題では，ここで作ったプログラムを活用せよ．

かかり受け解析の読み方
* 0 17D 1/1 0.388993

* : かかり受け解析の行であることを示す
0 : 文節番号
17D : かかり先の文節番号
1/1 : 主辞/機能語の位置と任意の個数の素性列
0.388993 : 係り関係のスコア。係りやすさの度合で、一般に大きな値ほど係りやすい。

'''

from knock40 import Morph

class Chunks:
    def __init__(self, morphs, dst):
        self.morphs = morphs
        self.dst = dst
        self.srcs = []

def chunk_list(target):
    i = 0
    morphs = [] #*区切りの形態素解析
    chunks = [] #Chunksのリスト
    sentences = []#EOS区切りのリスト

    for line in target:
        line = line.strip()
        
        if line[0] == '*':
            if len(morphs) > 0:
                chunks.append(Chunks(morphs, dst))
                morphs = []
            dst = int(line.split(' ')[2].rstrip('D'))

        elif line != 'EOS':
            morphs.append(Morph(line))

        else:
            chunks.append(Chunks(morphs, dst))
            for a, chunk in enumerate(chunks):
                if chunk.dst == -1 or chunk.dst == None:
                    continue
                chunks[chunk.dst].srcs.append(a)
            sentences.append(chunks)
            morphs = []
            chunks = []
            dst = None
        
        i += 1
        if i == 150: break

    return sentences


if __name__ == '__main__':
    f = open('ai.ja.txt.parsed', 'r')
    ans = chunk_list(f)

    for chunk in ans[2]:
        for morph in vars(chunk)['morphs']:
            print(vars(morph))
        print('dst: {}'.format(vars(chunk)['dst']))
        print('srcs: {}\n'.format(vars(chunk)['srcs']))

    f.close()