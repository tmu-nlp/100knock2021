#形態素解析結果（neko.txt.mecab）を読み込むプログラムを実装せよ．
#ただし，各形態素は表層形（surface），基本形（base），品詞（pos），品詞細分類1（pos1）をキーとするマッピング型に格納し，
#1文を形態素（マッピング型）のリストとして表現せよ．第4章の残りの問題では，ここで作ったプログラムを活用せよ．

#\u3000 とは全角スペースの意

f = open('neko.txt.mecab', 'r')

def make_morpheme(target):
    morphs = []
    for line in target:
        if line != 'EOS\n':
            morpheme = line.split('\t') #表層とそれ以外に分ける
            if morpheme[0] == '\n' or morpheme[0] == '': #改行文字を飛ばす
                continue
            else:
                morpheme[1] = morpheme[1].split(',')
            morphs.append({'surface':morpheme[0], 'base':morpheme[1][6], 'pos':morpheme[1][0], 'pos1':morpheme[1][1]})
    return morphs

if __name__ == '__main__':
    m = make_morpheme(f) #一文を表示
    for i in range(30): #一文が収まるぐらいの適当な単語数
        print(m[i])
        #if m[i]['surface'] == '。': break
    
    f.close()