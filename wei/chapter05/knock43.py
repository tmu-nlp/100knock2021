'''43. 名詞を含む文節が動詞を含む文節に係るものを抽出
名詞を含む文節が，動詞を含む文節に係るとき，これらをタブ区切り形式で抽出せよ．
ただし，句読点などの記号は出力しないようにせよ．'''


from knock41 import load_chunk



if __name__ ==  '__main__':
    filepath = './data/ai.ja/ai.ja.txt.parsed'
    res = load_chunk(filepath)
    for chunk in res[2].chunks:
        if int(chunk.dst) != -1:            # 文末以外の場合
            modifier = ''.join([morph.surface if morph.pos != '記号' else '' for morph in chunk.morphs])
            modifier_pos = [morph.pos for morph in chunk.morphs]
            modifiee = ''.join([morph.surface if morph.pos != '記号' else '' for morph in res[2].chunks[int(chunk.dst)].morphs])
            modifiee_pos = [morph.pos for morph in res[2].chunks[int(chunk.dst)].morphs]
            if '名詞' in modifier_pos and '動詞' in modifiee_pos:
                print(modifier,'\t', modifiee)
