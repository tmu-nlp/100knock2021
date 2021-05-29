'''
-- knock41 --
40に加えて，文節を表すクラスChunkを実装せよ．
このクラスは形態素（Morphオブジェクト）のリスト（morphs），係り先文節インデックス番号（dst），
係り元文節インデックス番号のリスト（srcs）をメンバ変数に持つこととする．
さらに，入力テキストの係り受け解析結果を読み込み，１文をChunkオブジェクトのリストとして表現し，
冒頭の説明文の文節の文字列と係り先を表示せよ．
本章の残りの問題では，ここで作ったプログラムを活用せよ．
'''

'''
-- *で始まる（係り受けに関する）行の説明 --
e.g.) * 1 14D 2/3 0.746013

* -> *
文節番号 -> 1
係り先の文節番号(係り先なし:-1) -> 14D
主辞の形態素番号/機能語の形態素番号 -> 2/3
係り関係のスコア(大きい方が係りやすい) -> 0.746013
'''
from collections import defaultdict
import knock40

class Chunk():
    def __init__(self, dst, srcs):
        self.morphs = []
        self.dst = dst
        self.srcs = srcs

def get_chunk_sentences():
    '''
    sentences = [sentence 1, sentence 2, ...]
    sentence = [chunk 1, cuhnk 2, ...]
    '''
    sentence = []
    sentences = []
    d1 = defaultdict(lambda:[])
    f = knock40.load_file()
    for line in f:
        
        # 文末まできたら、sentenceをsentencesに加え、sentenceとd1を再初期化。
        if line.strip() == "EOS":
            sentences.append(sentence)
            sentence = []
            d1 = defaultdict(lambda:[])

        # 文節情報の行なら、
        elif line.strip("\n").split(" ")[0] == "*":
            line = line.strip("\n").split(" ")
            chunk = Chunk(line[2][:-1],d1[line[1]])
            d1[line[2][:-1]].append(line[1])

            sentence.append(chunk)

        # 形態素の行なら、
        else :
            chunk.morphs.append(knock40.get_mor(line))

    return sentences
            
        
if __name__ == "__main__":
    sentences = get_chunk_sentences()
    for cnt, i in enumerate(sentences[0]):
        print(f"文節:{cnt}, 係り先:{i.dst}（, 係り元:",end="")
        if i.srcs:
            print(",".join(i.srcs),"）")
        else:
            print("-1）")
        for j in i.morphs:
            print(j.__dict__)
        print()

