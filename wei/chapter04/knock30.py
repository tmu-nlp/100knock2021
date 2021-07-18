'''30. 形態素解析結果の読み込み
形態素解析結果（neko.txt.mecab）を読み込むプログラムを実装せよ．
ただし，各形態素は表層形（surface），基本形（base），品詞（pos），品詞細分類1（pos1）をキーとするマッピング型に格納し，
1文を形態素（マッピング型）のリストとして表現せよ．第4章の残りの問題では，ここで作ったプログラムを活用せよ．

[Ref]
- MeCab の出力フォーマット
    - https://taku910.github.io/mecab/
        - 表層形\t品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用型,活用形,原形,読み,発音'''


def load_mecabf(mecabfile):
    sentences =[]
    morphs = []
    with open(mecabfile,'r',encoding='utf-8') as mecab_neko:
        for line in mecab_neko:
            if line != 'EOS\n':
                pair = line.split('\t')
                if len(pair) != 2 or pair[0] == ' ':      # 文頭以外の空白と改行文字はスキップ
                    continue
                else:
                    attr = pair[1].split(',')
                    morph = {'surface':pair[0], 'base': attr[6], 'pos': attr[0], 'pos1': attr[1]}
                    morphs.append(morph)
            else:
                sentences.append(morphs)            # センテンス毎の形態素リストを文リストに追加
                morphs = []
    return sentences

if __name__ == '__main__':

    mecabfile = './data/neko.txt.mecab'
    nekodata = load_mecabf(mecabfile)
    for morphs in nekodata[3]:
        print(morphs)

'''{'surface': '名前', 'base': '名前', 'pos': '名詞', 'pos1': '一般'}
{'surface': 'は', 'base': 'は', 'pos': '助詞', 'pos1': '係助詞'}
{'surface': 'まだ', 'base': 'まだ', 'pos': '副詞', 'pos1': '助詞類接続'}
{'surface': '無い', 'base': '無い', 'pos': '形容詞', 'pos1': '自立'}
{'surface': '。', 'base': '。', 'pos': '記号', 'pos1': '句点'}'''



