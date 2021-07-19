""'''
48. 名詞から根へのパスを抽出
文中のすべての名詞を含む文節に対し，その文節から構文木の根に至るパスを抽出せよ．
仕様：
各文節は表層形の形態素列で表現する。パスの開始から終了まで、各文節の表現を’->’で連結'''
from knock41 import load_chunk


if __name__ ==  '__main__':
    filepath = './data/ai.ja/ai.ja.txt.parsed'
    res = load_chunk(filepath)
    res = res[2]
    for chunk in res.chunks:
        if '名詞' in [morph.pos for morph in chunk.morphs]:
            path = [''.join(morph.surface for morph in chunk.morphs if morph.pos != '記号')]
            while chunk.dst != -1:
                # 名詞を含むchunkを先頭に、dstを根まで順に辿ってリストに追加
                path.append(''.join(morph.surface for morph in res.chunks[chunk.dst].morphs if morph.pos != '記号'))
                chunk = res.chunks[chunk.dst]
            print('->'.join(path))


'''
5パスを取り上げる:
人工知能->語->研究分野とも->される
じんこうちのう->語->研究分野とも->される
AI->エーアイとは->語->研究分野とも->される
エーアイとは->語->研究分野とも->される
計算->という->道具を->用いて->研究する->計算機科学->の->一分野を->指す->語->研究分野とも->される'''
