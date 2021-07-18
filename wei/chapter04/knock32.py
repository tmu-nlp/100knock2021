'''32. 動詞の基本形
動詞の基本形をすべて抽出せよ．'''

from knock30 import load_mecabf


def find_verbBase(nekodata, pos):
    result = set()
    for sentence in nekodata:
        for word in sentence:
            if word['pos'] == pos:
                result.add(word['base'])
    return result



if __name__ == '__main__':

    mecabfile = './data/neko.txt.mecab'
    nekodata = load_mecabf(mecabfile)
    bases = find_verbBase(nekodata, '動詞')
    print(f'動詞の原型の種類:{len(bases)}\n')
    print('---samples---')
    for k, v in enumerate(list(bases)[:10]):
        print(k, '\t', v)


'''
動詞の原型の種類:2300

---samples---
0 	 いざる
1 	 産まれる
2 	 潰れる
3 	 恐れ入る
4 	 遊ばす
5 	 切る
6 	 わる
7 	 形づくる
8 	 書き流す
9 	 とりのける'''