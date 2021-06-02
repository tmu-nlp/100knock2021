'''41. 係り受け解析結果の読み込み（文節・係り受け）
40に加えて，文節を表すクラスChunkを実装せよ．
このクラスは形態素（Morphオブジェクト）のリスト（morphs），係り先文節インデックス番号（dst），係り元文節インデックス番号のリスト（srcs）をメンバ変数に持つこととする．
さらに，入力テキストの係り受け解析結果を読み込み，１文をChunkオブジェクトのリストとして表現し，冒頭の説明文の文節の文字列と係り先を表示せよ．
本章の残りの問題では，ここで作ったプログラムを活用せよ'''

from knock40 import Morph



class Chunk():
    def __init__(self, morphs, dst):
        self.morphs = morphs        # 形態素(Morphオブジェクト)のリスト
        self.dst = dst              # 係り先文節インデックス番号
        self.srcs = []              # 係り元文節インデックス番号のリスト


def parse_cabocha(block):
    def check_create_chunk(tmp):
        if len(tmp) > 0:            # *から行ではないとき、tmpは1文のMorphオブジェクトリスト
            c = Chunk(tmp, dst)
            res.append(c)           # 1文のChunkオブジェクトのリスト
            tmp = []
        return tmp

    res = []
    tmp = []
    dst = []
    for line in block.split('\n'):
        if line =='':
            tmp = check_create_chunk(tmp)
        elif line[0] == '*':
            dst = line.split(' ')[2].rstrip('D')                 # 係り先文節インデックス番号を取得
            tmp = check_create_chunk(tmp)
        else:
            (surface, attr) =line.split('\t')                    # 1文のMorphオブジェクトリストを取得
            attr = attr.split(',')
            lineDict = {
                'surface':surface,
                'base':attr[6],
                'pos':attr[0],
                'pos1':attr[1]
            }
            tmp.append(Morph(lineDict))


    for i,r in enumerate(res):
        res[int(r.dst)].srcs.append(i)               # 係り元文節インデックス番号のリスト
    return res

if __name__ == '__main__':
    with open('./data/ai.ja/ai.ja.txt.parsed','r',encoding='utf-8') as f:
        blocks = f.read().split('EOS\n')
    blocks = list(filter(lambda x: x != '', blocks))
    blocks = [parse_cabocha(block) for block in blocks]
    for m in blocks[2]:
        print(''.join([mo.surface for mo in m.morphs]), m.dst, m.srcs)
        # 1文の各文節str、係り先文節のインデックス番号、係り元文節インデックス番号リスト


