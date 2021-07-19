''"""40. 係り受け解析結果の読み込み（形態素）
形態素を表すクラスMorphを実装せよ．このクラスは表層形（surface），基本形（base），
品詞（pos），品詞細分類1（pos1）をメンバ変数に持つこととする．
さらに，係り受け解析の結果（ai.ja.txt.parsed）を読み込み，
各文をMorphオブジェクトのリストとして表現し，冒頭の説明文の形態素列を表示せよ．"""


class Morph:
    '''形態素クラス
    　　表層形（surface），基本形（base），品詞（pos），品詞細分類1（pos1）をメンバ変数に持つ'''
    def __init__(self, morphs):
        surface, attr = morphs.split('\t')
        attr = attr.split(',')
        self.surface = surface
        self.base = attr[6]
        self.pos = attr[0]
        self.pos1 = attr[1]


def load_morphs(file_path):                          # 1文のMorphオブジェクトリストを作る
    sentences = []
    morphs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line[0] == '*':              # 係り受け関係を示す行をスキップ
                continue
            elif line != 'EOS\n':           # 文末以外の場合、Morph()で得た形態素辞書をリストに追加
                morphs.append(Morph(line))
            else:
                sentences.append(morphs)      # 文ごとに形態素辞書リストを追加
        return sentences


if __name__ ==  '__main__':
    filepath = './data/ai.ja/ai.ja.txt.parsed'
    res = load_morphs(filepath)
    for m in res[2]:

        print(vars(m))                 # 1文のMorphオブジェクトリスト

        '''vars([object])返回对象object的属性和属性值的字典对象,
            即返回对象的__dict__属性，常见有模块，类，实例。'''

