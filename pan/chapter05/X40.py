#形態素を表すクラスMorphを実装せよ
#このクラスは表層形（surface），基本形（base），品詞（pos），品詞細分類1（pos1）をメンバ変数に持つこととする
#さらに，係り受け解析の結果（ai.ja.txt.parsed）を読み込み，各文をMorphオブジェクトのリストとして表現し，冒頭の説明文の形態素列を表示せよ
import re

# Morphクラス
class Morph(object):   
    def __init__(self, line):
        self.surface = line[0]
        self.base = line[7]
        self.pos = line[1]
        self.pos1 = line[2]

# Morphオブジェクトのリスト（文単位）を返す関数
def analyze_morph(fname):    
    with open(fname, 'r', encoding='utf-8') as f:
        text = f.read().splitlines()
        text = [re.split('[\t, ]', t) for t in text]    #「\t」、「,」（形態素解析の行）、空白（係り受け解析の行）で分割
        morphs = []                                     #Morphオブジェクトのリスト（文単位）
        temp = []                                       #オブジェクトを文単位でまとめるため一時保管
        
        for line in text:                               #係り受け解析の行を除外
            if line[0] == '*' and len(line) == 5:
                continue
            elif line[0] != 'EOS':                      #形態素解析の行
                temp.append(Morph(line))   
            else:                                       #EOSの行（１文の終わり） 
                morphs.append(temp)
                temp = []
                                
    return morphs

# 3文目のMorphオブジェクトのリスト
ai_morphs = analyze_morph('/users/kcnco/github/100knock2021/pan/chapter05/ai.ja1.txt.parsed')[0]
print('surface\tbase\tpos\tpos1')

# Morphオブジェクトごとに
for ai_morph in ai_morphs:
    print('{}\t{}\t{}\t{}'.format(ai_morph.surface,     #メンバ変数を表示
                                  ai_morph.base, 
                                  ai_morph.pos, 
                                  ai_morph.pos1))
