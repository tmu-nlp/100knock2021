#形態素を表すクラスMorphを実装せよ
#このクラスは表層形（surface），基本形（base），品詞（pos），品詞細分類1（pos1）をメンバ変数に持つこととする
#さらに，係り受け解析の結果（ai.ja.txt.parsed）を読み込み，各文をMorphオブジェクトのリストとして表現し，冒頭の説明文の形態素列を表示せよ


class Morph():
    def __init__(self, line):
        line = line.split('\t')
        if line[0][0] == '*':
            self.meta = line[0].strip().split()
            self.EOS = False
        elif len(line) > 1:
            attr = line[1].split(',')
            self.surface = line[0]
            self.base = attr[-3]
            self.pos = attr[0]
            self.pos1 = attr[1]
            self.EOS = False
            self.meta = False
        else:
            self.EOS = True
            self.meta = False

    def show(self):
        if self.EOS:
            return 'EOS'
        elif not self.meta:
            return self.surface
        else:
            return self.meta

def morph2sents(listcabocha):
    sents = []
    sent = []
    for elem in listcabocha:
        if not elem.EOS:
            sent.append(elem)
        else:
            if sent != []:
                sents.append(sent)
            sent = []
    return sents


if __name__ == '__main__':
    with open('/users/kcnco/github/100knock2021/pan/chapter05/ai.ja1.txt.parsed','r') as ai:
        ai = ai.readlines()

    ai_morphs = []
    for i in range(len(ai)):
        ai_morphs.append(Morph(ai[i]))
        
    sents = morph2sents(ai_morphs)

    for sent in sents:
        for morph in sent:
            print(morph.show())
