'''
-- knock43 --
名詞を含む文節が，動詞を含む文節に係るとき，これらをタブ区切り形式で抽出せよ．
ただし，句読点などの記号は出力しないようにせよ．
'''
from knock41 import get_chunk_sentences

sentences = get_chunk_sentences()
for sentence in sentences:
    for chunk in sentence:
        if chunk.dst == "-1":
            continue
        if "名詞" in [a_mor.pos for a_mor in chunk.morphs] and \
           "動詞" in [a_mor.pos for a_mor in sentence[int(chunk.dst)].morphs]:
            s1 = "".join([a_mor.surface for a_mor in chunk.morphs if a_mor.pos != "記号"])
            s2 = "".join([a_mor.surface for a_mor in sentence[int(chunk.dst)].morphs if a_mor.pos != "記号"])
            print(f"{s1}\t{s2}")

