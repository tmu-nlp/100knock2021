'''
形態素を表すクラスMorphを実装せよ．
このクラスは表層形（surface），基本形（base），品詞（pos），品詞細分類1（pos1）
をメンバ変数に持つこととする．
さらに，係り受け解析の結果（ai.ja.txt.parsed）を読み込み，
各文をMorphオブジェクトのリストとして表現し，冒頭の説明文の形態素列を表示せよ．
'''
'''
1. *
2. 文節番号
3. 係り先の文節番号(係り先なし:-1)
4. 主辞の形態素番号/機能語の形態素番号
5. 係り関係のスコア(大きい方が係りやすい)
'''

from collections import defaultdict


class Morph:
    def __init__(self, surface, base , pos, pos1):
        self.surface = surface
        self.base = base
        self.pos = pos
        self.pos1 = pos1
    
    def __str__(self):
        return \
            f"表層形:{self.surface}\t \
                基本形:{self.base}\t \
                    品詞:{self.pos}\t \
                        品詞細分類1:{self.pos1}"


with open('ai.ja.txt.parsed') as lines:
    setences, morphs = [], []
    for line in lines:
        if line[0] == "*": continue
        elif line == 'EOS\n':
            setences.append(morphs)
            morphs = []
        else:
            surface, info = line.rstrip().split('\t')
            info = info.split(',')
            base = info[-3]
            pos = info[0]
            pos1 = info[1]
            morphs.append(Morph(surface, base, pos, pos1))

if __name__ == "__main__":
    for m in setences[2]:
        '''
        varsと__dict__
        '''
        print(m.__dict__)
