#形態素解析結果（neko.txt.mecab）を読み込むプログラムを実装せよ．
#ただし，各形態素は表層形（surface），基本形（base），品詞（pos），品詞細分類1（pos1）をキーとするマッピング型に格納し，
#1文を形態素（マッピング型）のリストとして表現せよ．
import re

def mecab(file):
    with open(file, encoding='utf-8') as data:
        dicts = []
        line = data.readline()
        while(line):
            result = re.split('[,\t\n]', line)
            result = result[:-1]
            line = data.readline()
            if len(result) < 2:
                continue
            dict = {'surface': result[0],
                    'base': result[7],
                    'pos': result[1],
                    'pos1': result[2],}
            dicts.append(dict)
            if result[0] == '。':
                yield dicts
                dicts = []

if __name__ == '__main__':
    file = "/users/kcnco/github/100knock2021/pan/chapter04/neko.txt.mecab"
    lines = mecab(file)
    t = 0
    for line in lines: 
        print(line)
        t+=1
        if t == 30:
            break
