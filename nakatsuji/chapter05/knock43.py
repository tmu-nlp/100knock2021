'''
名詞を含む文節が，動詞を含む文節に係るとき，
これらをタブ区切り形式で抽出せよ．
ただし，句読点などの記号は出力しないようにせよ．
'''
from knock41 import sentences
from knock42 import words
N2Vs = []
sentence = sentences[2]

for i, chunk in enumerate(sentence):
    morphs = chunk.morphs
    for morph in morphs:
        flag = 0
        if morph.pos == '名詞':
            for v_morph in sentence[chunk.dst].morphs:
                if v_morph.pos == '動詞':
                    N2Vs.append(f'{words[i]}\t{words[chunk.dst]}')
                    flag = 1
                    break
        if flag: break
if __name__ == "__main__":
    for n2v in N2Vs:
        print(n2v)

'''
道具を  用いて
知能を  研究する
一分野を        指す
知的行動を      代わって
人間に  代わって
コンピューターに        行わせる
研究分野とも    される
'''