'''41. 係り受け解析結果の読み込み（文節・係り受け）
40に加えて，文節を表すクラスChunkを実装せよ．
このクラスは形態素（Morphオブジェクト）のリスト（morphs），係り先文節インデックス番号（dst），係り元文節インデックス番号のリスト（srcs）をメンバ変数に持つこととする．
さらに，入力テキストの係り受け解析結果を読み込み，１文をChunkオブジェクトのリストとして表現し，冒頭の説明文の文節の文字列と係り先を表示せよ．
本章の残りの問題では，ここで作ったプログラムを活用せよ'''

from knock40 import Morph

# 文は文節リストを要素に持ち、文節は形態素リストを要素に持つ
class Chunk():
    def __init__(self, morphs, dst):
        self.morphs = morphs        # 形態素(Morphオブジェクト)のリスト
        self.dst = dst              # 係り先文節インデックス番号
        self.srcs = []              # 係り元文節インデックス番号のリスト


class Sentence():
    def __init__(self, chunks):
        self.chunks = chunks
        for i, chunk in enumerate(self.chunks):
            if chunk.dst != -1:
                self.chunks[chunk.dst].srcs.append(i)

def load_chunk(file_path):
    sentences = []
    chunks = []
    morphs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line[0] == '*':                             # * 0 -1D 1/1 0.000000　
                if len(morphs) > 0:
                    chunks.append(Chunk(morphs, dst))       # 直前の文節の情報に対して、Chunk()による結果を文節リストに追加
                    morphs = []
                dst = int(line.split(' ')[2].rstrip('D'))   # 直後の文節の係り先を取得　　
            elif line != 'EOS\n':
                morphs.append(Morph(line))       # Morph objectを返し、形態素リストに追加
            else:
                chunks.append(Chunk(morphs, dst))
                sentences.append(Sentence(chunks))      #文節リストにSentence()を適用し、文リストに追加
                morphs = []
                chunks = []
                dst = int()
    return sentences


if __name__ ==  '__main__':
    filepath = './data/ai.ja/ai.ja.txt.parsed'
    res = load_chunk(filepath)
    for chunk in res[2].chunks:
        print([morph.surface for morph in chunk.morphs], chunk.dst, chunk.srcs)
