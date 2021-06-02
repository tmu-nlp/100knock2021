''"""40. 係り受け解析結果の読み込み（形態素）
形態素を表すクラスMorphを実装せよ．このクラスは表層形（surface），基本形（base），
品詞（pos），品詞細分類1（pos1）をメンバ変数に持つこととする．
さらに，係り受け解析の結果（ai.ja.txt.parsed）を読み込み，
各文をMorphオブジェクトのリストとして表現し，冒頭の説明文の形態素列を表示せよ．"""


class Morph():
    '''形態素クラス
    　　表層形（surface），基本形（base），品詞（pos），品詞細分類1（pos1）をメンバ変数に持つ'''
    def __init__(self, dc):
        self.surface = dc['surface']
        self.base = dc['base']
        self.pos = dc['pos']
        self.pos1 = dc['pos1']


def parse_cabocha(block):                          # 1文のMorphオブジェクトリストを作る
    res = []
    for line in block.split('\n'):                  # list中存在''元素
        if line == '':
            return res
        elif line[0] == '*':
            continue
        # print(line.split('\t'))
        (surface, attr) = line.split('\t')
        '''此处遇到ValueError(expected2,got1),原因是：psrsed文件中最后不是'EOS\n',
            导致最后一行的'EOS'被遍历,len该行为1，无法得到2个值。
            解决方法：psrsed文件中加一行'''
        attr = attr.split(',')
        lineDict = {
            'surface':surface,
            'base':attr[6],
            'pos':attr[0],
            'pos1':attr[1]
        }
        res.append(Morph(lineDict))
    return res


if __name__ ==  '__main__':
    with open('./data/ai.ja/ai.ja.txt.parsed','r',encoding='utf-8') as f:
        blocks = f.read().split('EOS\n')                          # parsed文件中两个‘EOS\n’的部分将成为blocks列表中的''，共74个。
        # print(len(blocks))                                      # 文件ai.ja有158行
    blocks = list(filter(lambda x: x !='', blocks))               # 过滤掉列表中的''后, len(blocks)-> 83文
    blocks = [parse_cabocha(block) for block in blocks]           # 各文をMorphオブジェクトリストとし表現




    for m in blocks[2]:

        print(vars(m))                 # 1文のMorphオブジェクトリスト

        '''vars([object])返回对象object的属性和属性值的字典对象,
            即返回对象的__dict__属性，常见有模块，类，实例。'''

