'''
係り元の文節と係り先の文節のテキストをタブ区切り形式ですべて抽出せよ．
ただし，句読点などの記号は出力しないようにせよ．
'''
from knock41 import sentences
dependencies = []

for sentence in sentences:
    dependency = []
    words = []
    for chunk in sentence:
        morphs = chunk.morphs
        word = ''
        for morph in morphs:
            if morph.pos == '記号':
                continue
            else: word += morph.surface
        words.append(word)
    for i, chunk in enumerate(sentence):
        if chunk.dst != -1:
            dependency.append(f'{words[i]}\t{words[chunk.dst]}')
    dependencies.append(dependency)

    
if __name__ == "__main__":
    for d in dependencies[2]:
        print(d)

'''
人工知能        語
じんこうちのう  語
AI      エーアイとは
エーアイとは    語
計算    という
という  道具を
概念と  道具を
コンピュータ    という
という  道具を
.
.
.
'''