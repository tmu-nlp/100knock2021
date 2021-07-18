'''31. 動詞
動詞の表層形をすべて抽出せよ．
'''""

from knock30 import load_mecabf


def find_verb(nekodata, pos):
    result = set()
    for sentence in nekodata:
        for word in sentence:
            if word['pos'] == pos:
                result.add(word['surface'])
    return result



if __name__ == '__main__':

    mecabfile = './data/neko.txt.mecab'
    nekodata = load_mecabf(mecabfile)
    surfaces = find_verb(nekodata, '動詞')
    print(f'動詞の表層形の種類:{len(surfaces)}\n')
    print('---samples---')
    for k, v in enumerate(list(surfaces)[:10]):
        print(k, '\t', v)

'''
動詞の表層形の種類:3893

---samples---
0 	 やめる
1 	 振り翳し
2 	 怠
3 	 着せる
4 	 開ける
5 	 なれ
6 	 よごれ
7 	 写っ
8 	 騒が
9 	 塞ぐ'''