'''
-- knock40 の問題 --
形態素を表すクラスMorphを実装せよ．
このクラスは表層形（surface），基本形（base），品詞（pos），品詞細分類1（pos1）をメンバ変数に持つこととする．
さらに，係り受け解析の結果（ai.ja.txt.parsed）を読み込み，各文をMorphオブジェクトのリストとして表現し，冒頭の説明文の形態素列を表示せよ．
'''
'''
-- 単語の行の説明 --
表層形 （Tab区切り）
品詞
品詞細分類1
品詞細分類2
品詞細分類3
活用形
活用型
原形
読み
発音
'''

class Morph:
    def __init__(self, surface, base, pos, pos1):
        self.surface = surface
        self.base = base
        self.pos = pos
        self.pos1 = pos1

def load_file():
    with open("ai.ja/ai.ja.txt.parsed", "r") as f:
        return f.readlines()

def get_mor(line):
    surface, mor = line.split("\t")
    mor = mor.strip("\n").split(",")
    morph = Morph(surface, mor[6], mor[0], mor[1])
    return morph

def get_mor_sentences():
    f = load_file()
    sentence = []
    sentences = []
    for line in f:
        if line.strip() == "EOS":
            sentences.append(sentence)
            sentence = []
        elif "\t" in line:
            sentence.append(get_mor(line))
    return sentences

if __name__ == "__main__":
    sentences = get_mor_sentences()
    for i in sentences[2]:
        print(i.__dict__)
