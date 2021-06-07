'''31. 動詞
動詞の表層形をすべて抽出せよ．

'''

from knock30 import load_mecabf


def find_verb(nekodata, pos):
    result = []
    for sentence in nekodata:
        for word in sentence:
            if word['pos'] == pos:
                result.append(word['surface'])
    return result



if __name__ == '__main__':

    mecabfile = './data/neko.txt.mecab'
    nekodata = load_mecabf(mecabfile)
    surface = find_verb(nekodata, '動詞')
    print(surface)