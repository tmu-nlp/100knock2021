#形態素を表すクラスMorphを実装せよ．このクラスは表層形（surface），基本形（base），品詞（pos），品詞細分類1（pos1）
#をメンバ変数に持つこととする．さらに，係り受け解析の結果（ai.ja.txt.parsed）を読み込み，各文をMorphオブジェクトのリストとして表現し，
#冒頭の説明文の形態素列を表示せよ．

class Morph:
    def __init__(self, line):
        line = line.strip()
        l = line.split('\t')
        l[1] = l[1].split(',')
        self.surface = l[0]
        self.base = l[1][6]
        self.pos = l[1][0]
        self.pos1 = l[1][1]

if __name__ == '__main__':
    f = open('ai.ja.txt.parsed', 'r')
    morph = []#１文の形態素解析のリスト
    sentences = []#morphのリスト

    for line in f:
        if '\t' in line:
            #形態素解析できるものはmorphに追加
            morph.append(Morph(line))
        elif 'EOS' in line:
            #文が終わったらsentencesにmorphを追加してmorphを初期化
            sentences.append(morph)
            morph = []
    
    ans = sentences[2]

    for mo in ans:
        print(vars(mo))

    f.close()
