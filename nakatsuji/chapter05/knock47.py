'''
動詞のヲ格にサ変接続名詞が入っている場合のみに着目したい．
46のプログラムを以下の仕様を満たすように改変せよ．

-「サ変接続名詞+を（助詞）」で構成される文節が動詞に係る場合のみを対象とする
-述語は「サ変接続名詞+を+動詞の基本形」とし，文節中に複数の動詞があるときは，
    最左の動詞を用いる
-述語に係る助詞（文節）が複数あるときは，すべての助詞をスペース区切りで辞書順に並べる
-述語に係る文節が複数ある場合は，すべての項をスペース区切りで並べる（助詞の並び順と揃えよ）

例えば「また、自らの経験を元に学習を行う強化学習という手法もある。」という文から，
以下の出力が得られるはずである

    学習を行う	に を	元に 経験を
'''
from typing import DefaultDict
from knock41 import sentences
from knock42 import dependencies
from collections import *
pattern_V_case = []
cnt = 0
with open('output47.txt', 'w') as out:
    for sentence in sentences:
        
        for i, chunk in enumerate(sentence):
            verb = ''
            for morph in chunk.morphs:
                if morph.pos == '動詞':
                    verb = morph.base
                    break
            

            if verb == '':
                continue
            #さ変＋をがあるか
            predicate = ''
            for noun, wo in zip(sentence[i-1].morphs,sentence[i-1].morphs[1:]):
                if noun.pos1 == 'サ変接続' and wo.pos == '助詞' and wo.surface == 'を':
                    predicate = noun.surface + wo.surface + verb
                    cnt += 1
                    break
            cases_terms = defaultdict(str)
            if predicate == '':
                continue
            for src in chunk.srcs:
                
                case = ''
                term = ''
                for morph in sentence[src].morphs:
                    if not morph.pos == '記号':
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
            writing = f'{predicate}\t{cases}\t{terms}'
            #print(writing)
            out.write(writing + '\n')
            
print(cnt)
        