'''
-- knock42 --
係り元の文節と係り先の文節のテキストをタブ区切り形式ですべて抽出せよ．
ただし，句読点などの記号は出力しないようにせよ．
'''
from knock41 import get_chunk_sentences

sentences = get_chunk_sentences()
for sentence in sentences:
    for chunk in sentence:
        if chunk.dst == "-1":
            print("".join([a_mor.surface for a_mor in chunk.morphs if a_mor.pos != "記号"]))
        else:
            s1 = "".join([a_mor.surface for a_mor in chunk.morphs if a_mor.pos != "記号"])
            s2 = "".join([a_mor.surface for a_mor in sentence[int(chunk.dst)].morphs if a_mor.pos != "記号"])
            print(f"{s1}\t{s2}")
