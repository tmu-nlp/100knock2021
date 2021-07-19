""'''
49. 名詞間の係り受けパスの抽出
文中のすべての名詞句のペアを結ぶ最短係り受けパスを抽出せよ
仕様：
i -> a -> b -> j -> 根 であれば、 i -> a -> b -> j
i -> a -> k -> 根、j -> b -> k -> 根 であれば、i -> a | j -> b | k'''

from itertools import combinations
import re


from knock41 import load_chunk


if __name__ ==  '__main__':
    filepath = './data/ai.ja/ai.ja.txt.parsed'
    res = load_chunk(filepath)
    res = res[2]
    nouns = []
    for i, chunk in enumerate(res.chunks):
        if '名詞' in [morph.pos for morph in chunk.morphs]:
            nouns.append(i)
    for i, j in combinations(nouns, 2):
        path_i = []
        path_j = []
        while i != j:
            if i < j:
                path_i.append(i)
                i = res.chunks[i].dst
            else:
                path_j.append(j)
                j = res.chunks[j].dst
        if len(path_j) == 0:
            chunk_X = ''.join([morph.surface if morph.pos != '名詞' else 'X' for morph in res.chunks[path_i[0]].morphs])
            chunk_Y = ''.join([morph.surface if morph.pos != '名詞' else 'Y' for morph in res.chunks[i].morphs])

            chunk_X = re.sub('X+', 'X', chunk_X)
            chunk_Y = re.sub('Y+', 'Y', chunk_Y)
            path_X2Y = [chunk_X] + [''.join(morph.surface for morph in res.chunks[n].morphs) for n in path_i[1:]] + [chunk_Y]

            print(' -> '.join(path_X2Y))
        else:
            chunk_X = ''.join([morph.surface if morph.pos != '名詞' else 'X' for morph in res.chunks[path_i[0]].morphs])
            chunk_Y = ''.join([morph.surface if morph.pos != '名詞' else 'Y' for morph in res.chunks[path_j[0]].morphs])
            chunk_k = ''.join([morph.surface for morph in res.chunks[i].morphs])
            chunk_X = re.sub('X+', 'X', chunk_X)
            chunk_Y = re.sub('Y+', 'Y', chunk_Y)
            path_X = [chunk_X] + [''.join(morph.surface for morph in res.chunks[n].morphs) for n in path_i[1:]]
            path_Y = [chunk_Y] + [''.join(morph.surface for morph in res.chunks[n].morphs) for n in path_j[1:]]
            print(' | '.join(['->'.join(path_X), ' -> '.join(path_Y), chunk_k]))


