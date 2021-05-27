'''
45のプログラムを改変し，述語と格パターンに続けて項
（述語に係っている文節そのもの）
をタブ区切り形式で出力せよ．
45の仕様に加えて，以下の仕様を満たすようにせよ．

項は述語に係っている文節の単語列とする
（末尾の助詞を取り除く必要はない）
述語に係る文節が複数あるときは，
助詞と同一の基準・順序でスペース区切りで並べる
「ジョン・マッカーシーはAIに関する最初の会議で人工知能
という用語を作り出した。」
という例文を考える． 
この文は「作り出す」という１つの動詞を含み，
「作り出す」に係る文節は「ジョン・マッカーシーは」，「会議で」，
「用語を」であると解析された場合は，次のような出力になるはずである．

作り出す	で は を	会議で ジョンマッカーシーは 用語を
'''

from typing import DefaultDict
from knock41 import sentences
from knock42 import dependencies
from collections import *
pattern_V_case = []
with open('output46.txt', 'w') as out:
    for sentence in sentences:
        #係り先が動詞かの否かのフラグ
        flag = 0


        for chunk in sentence:
            verb = ''
            for morph in chunk.morphs:
                if morph.pos == '動詞':
                    verb = morph.base
                    break
            if verb == '':
                continue
            cases_terms = defaultdict(str)
            for src in chunk.srcs:
                
                case = ''
                term = ''
                for morph in sentence[src].morphs:
                    term += morph.surface
                    if morph.pos == '助詞':
                        case += morph.base
                cases_terms[term] = case
            sorted(cases_terms.items(), key=lambda x: x[1])
            cases, terms = [], []
            for term, case in cases_terms.items():
                cases.append(case)
                terms.append(term)
            cases, terms = ' '.join(cases), ' '.join(terms)
            writing = f'{verb}\t{cases}\t{terms}' 
            out.write(writing + '\n')
        