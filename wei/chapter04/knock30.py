'''30. 形態素解析結果の読み込み
形態素解析結果（neko.txt.mecab）を読み込むプログラムを実装せよ．
ただし，各形態素は表層形（surface），基本形（base），品詞（pos），品詞細分類1（pos1）をキーとするマッピング型に格納し，
1文を形態素（マッピング型）のリストとして表現せよ．第4章の残りの問題では，ここで作ったプログラムを活用せよ．

[Ref]
- MeCab の出力フォーマット
    - https://taku910.github.io/mecab/
        - 表層形\t品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用型,活用形,原形,読み,発音'''


import MeCab
import re

# def mecab_result(filename):
#     with open(filename, 'r') as neko,open('./data/neko.txt.mecab','w+') as outfile:
#         mecab = MeCab.Tagger('')
#         for line in neko:
#             result = mecab.parse(line)
#             print(result)
#             outfile.write(result)

def load_macab(mecabfile):
    with open(mecabfile,'r',encoding='utf-8') as mecabneko:
        morphenes = []
        line = mecabneko.readline()
        while(line):
            result = re.split('[,\t\n]',line)
            result = result[:-1]
            line = mecabneko.readline()
            if len(result)<2:
                continue
            morphene = {
                'surface': result[0],
                'base': result[7],
                'pos': result[1],
                'pos1': result[2],
            }
            morphenes.append(morphene)
            if result[0] == '。':
                yield morphenes
                morphenes = []

if __name__ == '__main__':
    # filepath = './data/neko.txt'
    mecabfile = './data/neko.txt.mecab'
    nekodata = load_macab(mecabfile)
    i = 0
    for line in nekodata:
        print(line)
        i += 1
        if i == 10:
            break




