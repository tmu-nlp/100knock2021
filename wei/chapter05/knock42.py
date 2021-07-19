'''42. 係り元と係り先の文節の表示
係り元の文節と係り先の文節のテキストをタブ区切り形式ですべて抽出せよ．
ただし，句読点などの記号は出力しないようにせよ．'''


from knock41 import load_chunk



if __name__ ==  '__main__':
    filepath = './data/ai.ja/ai.ja.txt.parsed'
    res = load_chunk(filepath)
    for chunk in res[2].chunks:
        if int(chunk.dst) != -1:
            modifier = ''.join([morph.surface if morph.pos != '記号' else '' for morph in chunk.morphs])
            modifiee = ''.join([morph.surface if morph.pos != '記号' else '' for morph in res[2].chunks[int(chunk.dst)].morphs])
            print(modifier,'\t', modfifiee)


