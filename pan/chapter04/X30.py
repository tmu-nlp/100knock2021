#形態素解析結果（neko.txt.mecab）を読み込むプログラムを実装せよ．
#ただし，各形態素は表層形（surface），基本形（base），品詞（pos），品詞細分類1（pos1）をキーとするマッピング型に格納し，
#1文を形態素（マッピング型）のリストとして表現せよ．

import re

def mecab(file):
    #データを読み込み 
    with open(file, encoding='utf-8') as data:
        #空のリストを用意して
        dicts = []
        #readline()を使って、一行ずつ読み込む
        line = data.readline()
        while(line): 
            #re.split()を使って、lineは各要素に区切られる
            result = re.split(r'[,\t\n]', line)
            result = result[:-1]
            line = data.readline()
            #最後のEOS\nの処理
            if len(result) < 2:
                continue
            #辞書作成、リストに追加
            dict = {'surface': result[0],  #表層(surface)：リスト[0]番目
                    'base': result[7],     #基本(base)：リスト[7]番目
                    'pos': result[1],      #品詞(pos)：リスト[1]番目
                    'pos1': result[2],}    #品詞細分類1(pos1)：リスト[2]番目            
            #形態素リストを文リストに追加し、そして用意したdictsに格納します
            dicts.append(dict)
            #pos1（品詞細分類1）が句点なら文の終わりと判定する
            if result[0] == '。':
                yield dicts
                #返す
                dicts = []

if __name__ == '__main__':
    file = "/users/kcnco/github/100knock2021/pan/chapter04/neko.txt.mecab"
    #一文ずつ辞書のリストを作成
    lines = mecab(file)
    t = 0
    for line in lines: 
        print(line)
        t+=1
        if t == 30:
            break
