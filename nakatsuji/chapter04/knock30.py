'''
形態素解析結果（neko.txt.mecab）を読み込むプログラムを実装せよ．
ただし，各形態素は表層形（surface），基本形（base），品詞（pos），品詞細分類1（pos1）
をキーとするマッピング型に格納し，
1文を形態素（マッピング型）のリストとして表現せよ．
第4章の残りの問題では，ここで作ったプログラムを活用せよ．
'''
'''
表層形\t品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用形,活用型,原形,読み,発音
'''



with open('neko.txt.mecab') as f:
    sentences = []
    a_sectence = []
    for line in f:
        if line == '\n': continue
        if line == 'EOS\n':
            sentences.append(a_sectence)
            a_sectence = []
        else:
            
            line = line.split('\t')
            sur = line[0]
            info = line[1].split(',')
            ##文頭と文末の空白削除
            if sur == '': continue
            if info[1] == '空白': continue
            word = {}
            word['surface'] = sur
            word['base'] = info[-3]
            word['pos'] = info[0]
            word['pos1'] = info[1]
            a_sectence.append(word)
if __name__ == "__main__":
    for i in sentences[2]:
        print(i)

    
