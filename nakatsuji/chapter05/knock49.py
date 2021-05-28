'''
文中のすべての名詞句のペアを結ぶ最短係り受けパスを抽出せよ. 
ただし，名詞句ペアの文節番号がiとj（i<j）のとき，
係り受けパスは以下の仕様を満たすものとする．

 - 問題48と同様に，パスは開始文節から終了文節に至るまでの
    各文節の表現（表層形の形態素列）を” -> “で連結して表現する
 - 文節iとjに含まれる名詞句はそれぞれ，XとYに置換する

また，係り受けパスの形状は，以下の2通りが考えられる．

 - 文節iから構文木の根に至る経路上に文節jが存在する場合: 
    文節iから文節jのパスを表示
 - 上記以外で，文節iと文節jから構文木の根に至る経路上で共通の文節kで交わる場合: 
    文節iから文節kに至る直前のパスと文節jから文節kに至る直前までのパス，
    文節kの内容を” | “で連結して表示

「ジョン・マッカーシーはAIに関する最初の会議で人工知能という用語を作り出した。」
    という例文を考える． CaboChaを係り受け解析に用いた場合，次のような出力が得られると思われる．

Xは | Yに関する -> 最初の -> 会議で | 作り出した
Xは | Yの -> 会議で | 作り出した
Xは | Yで | 作り出した
Xは | Yという -> 用語を | 作り出した
Xは | Yを | 作り出した
Xに関する -> Yの
Xに関する -> 最初の -> Yで
Xに関する -> 最初の -> 会議で | Yという -> 用語を | 作り出した
Xに関する -> 最初の -> 会議で | Yを | 作り出した
Xの -> Yで
Xの -> 会議で | Yという -> 用語を | 作り出した
Xの -> 会議で | Yを | 作り出した
Xで | Yという -> 用語を | 作り出した
Xで | Yを | 作り出した
Xという -> Yを
'''
from os import path
from knock41 import sentences
from knock42 import dependencies
from knock48 import paths
from collections import *
from itertools import *
import re

def main(sentence):
    #名詞句を格納
    
    nouns = [i for i, chunk in enumerate(sentence) if '名詞' in [morph.pos for morph in chunk.morphs]]

    for i, j in combinations(nouns, 2):  # 名詞を含む文節のペアごとにパスを作成
        path_i = []
        path_j = []
        while i != j:
          if i < j:
            path_i.append(i)
            i = sentence[i].dst
          else:
            path_j.append(j)
            j = sentence[j].dst
    
        if len(path_j) == 0:  # 1つ目のケース
            chunk_X = ''.join([morph.surface if morph.pos != '名詞' else 'X' for morph in sentence[path_i[0]].morphs])
            chunk_Y = ''.join([morph.surface if morph.pos != '名詞' else 'Y' for morph in sentence[i].morphs])
            chunk_X = re.sub('X+', 'X', chunk_X)
            chunk_Y = re.sub('Y+', 'Y', chunk_Y)
            path_XtoY = [chunk_X] + [''.join(morph.surface for morph in sentence[n].morphs) for n in path_i[1:]] + [chunk_Y]
            print(' -> '.join(path_XtoY))
        else:  # 2つ目のケース
            chunk_X = ''.join([morph.surface if morph.pos != '名詞' else 'X' for morph in sentence[path_i[0]].morphs])
            chunk_Y = ''.join([morph.surface if morph.pos != '名詞' else 'Y' for morph in sentence[path_j[0]].morphs])
            chunk_k = ''.join([morph.surface for morph in sentence[i].morphs])
            chunk_X = re.sub('X+', 'X', chunk_X)
            chunk_Y = re.sub('Y+', 'Y', chunk_Y)
            path_X = [chunk_X] + [''.join(morph.surface for morph in sentence[n].morphs) for n in path_i[1:]]
            path_Y = [chunk_Y] + [''.join(morph.surface for morph in sentence[n].morphs) for n in path_j[1:]]
            print(' | '.join([' -> '.join(path_X), ' -> '.join(path_Y), chunk_k]))


if __name__ == '__main__':
    main(sentences[1])