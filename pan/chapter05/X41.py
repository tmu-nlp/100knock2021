#40に加えて，文節を表すクラスChunkを実装せよ
#このクラスは形態素（Morphオブジェクト）のリスト（morphs），係り先文節インデックス番号（dst），係り元文節インデックス番号のリスト（srcs）をメンバ変数に持つこととする
#さらに，入力テキストの係り受け解析結果を読み込み，１文をChunkオブジェクトのリストとして表現し，冒頭の説明文の文節の文字列と係り先を表示せよ
import re
from X40 import Morph

#Chunkクラスを作成。問題文で指定されたメンバ変数の他に、文節単位で文字列を表示するためのインスタンスを追加。
class Chunk(object):
    def __init__(self, line):      
        self.morphs = []                                #形態素（Morphオブジェクト）のリスト
        self.dst = int(line[2].rstrip('D'))             #係り先文節インデックス番号のリスト
        self.srcs = []                                  #係り元文節インデックス番号のリスト
            
    def join_surface(self):                      
        return ''.join([morph.surface for morph in self.morphs])
                                                        #文節単位でsurfaceを結合

#Chunkオブジェクトのリスト（文単位）を返す関数
def analyze_chunk(fname):    
    with open(fname,'r', encoding='utf-8') as f:
        text = f.read().splitlines()
        text = [re.split('[\t, ]', t) for t in text]    #「\t」、「,」（形態素解析の行）、空白（係り受け解析の行）で分割
        chunks = []                                     #Chunkオブジェクトのリスト（文単位）
        temp = []                                       #オブジェクトを文単位でまとめるため一時保管
       
        for line in text:            
            if line[0] == '*' and len(line) == 5:       #係り受け解析の行
                chunk = Chunk(line)                     #Chunkオブジェクトを作成
                temp.append(chunk)                      #Chunkオブジェクトをtempに追加（この時点ではmorphsとsrcsは空リスト）
                  
            elif line[0] != 'EOS':                      #形態素解析の行   
                chunk.morphs.append(Morph(line))        #ChunkオブジェクトのmorphsにMorphオブジェクトを追加
                        
            else:                                       #EOSの行（１文の終わり）
                for i, chunk in enumerate(temp):        #Chunkオブジェクトの中で係り先があるもののみ
                    if chunk.dst != -1:
                        temp[chunk.dst].srcs.append(i)  #係り先のChunkオブジェクトのsrcsに係り元文節インデックスを追加                        
                chunks.append(temp)
                temp = []

    return chunks

#関数を実行して、文節の文字列と係り先、そして確認のために係り元も表示
ai_chunks = analyze_chunk('/users/kcnco/github/100knock2021/pan/chapter05/ai.ja1.txt.parsed')[0]
print('surface\tdst\tsrcs')

for ai_chunk in ai_chunks:
    print('{}\t{}\t{}'.format(ai_chunk.join_surface(),   #文節の文字列、係り先、係り元を表示
                              ai_chunk.dst, 
                              ai_chunk.srcs))

'''
print('surface\tbase\tpos\tpos1')
for ai_chunk in ai_chunks:
    for ai_morph in ai_chunk.morphs:                     #Morphオブジェクトごとに
        print('{}\t{}\t{}\t{}'.format(ai_morph.surface,  #メンバ変数を表示
                                      ai_morph.base, 
                                      ai_morph.pos, 
                                      ai_morph.pos1))
'''
