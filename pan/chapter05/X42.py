#係り元の文節と係り先の文節のテキストをタブ区切り形式ですべて抽出せよ．ただし，句読点などの記号は出力しないようにせよ．
import re
from X40 import Morph

class Chunk(object):
    def __init__(self, line):
        self.morphs = []                                         #形態素のリスト
        self.dst = int(line[2].rstrip('D'))                      #係り先文節インデックス番号のリスト
        self.srcs = []                                           #係り元文節インデックス番号のリスト
    
    def join_surface(self):                                      #文節単位でsurfaceを結合
        return ''.join([morph.surface for morph in self.morphs])

    def join_surface_womark(self):                               #文節単位でsurfaceを結合（記号以外）
        return ''.join([morph.surface for morph in self.morphs
                       if morph.pos != '記号'])

def analyze_chunk(fname):
    with open(fname, 'r') as f:
    #with open(fname,'r', 'encoding=utf-8') as f:
        text = f.read().splitlines() 
        text = [re.split('[\t, ]', t) for t in text]
        chunks = []
        temp = []
       
        for line in text:
            if line[0] == '*' and len(line) == 5:
                chunk = Chunk(line)
                temp.append(chunk) 
            elif line[0] != 'EOS':
                chunk.morphs.append(Morph(line))
            else:
                for i, chunk in enumerate(temp):
                    if chunk.dst != -1:
                        temp[chunk.dst].srcs.append(i)                       
                chunks.append(temp)
                temp = []
        
    return chunks

ai_chunks_sentences = analyze_chunk('/users/kcnco/github/100knock2021/pan/chapter05/ai.ja1.txt.parsed')[0:]
print('srcs\tdst')

for ai_chunks in ai_chunks_sentences:
    for ai_chunk in ai_chunks:
        #係り先がある場合
        if ai_chunk.dst != -1:
            #係り元: 文節単位でsurfaceを結合（記号以外）
            s = ai_chunk.join_surface_womark()
            #係り先: sの係り先の文節のsurfaceを結合（記号以外）
            d = ai_chunks[ai_chunk.dst].join_surface_womark()
            
            #係り元も係り先も記号のみではない場合
            if s != '' and d != '':
                print('{}\t{}'.format(s, d))
