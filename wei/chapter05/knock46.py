""'''
述語と格パターンに続けて,項（述語に係っている文節そのもの）をタブ区切り形式で出力せよ．
45の仕様に加えて，以下の仕様を満たすようにせよ:
項は述語に係っている文節の単語列とする（末尾の助詞を取り除く必要はない）
述語に係る文節が複数あるときは，助詞と同一の基準・順序でスペース区切りで並べる
'''


from knock41 import load_chunk


if __name__ ==  '__main__':
    filepath = './data/ai.ja/ai.ja.txt.parsed'
    res = load_chunk(filepath)
    with open('./ans46.txt','w', encoding='utf-8') as f:
        for sentence in res:
            for chunk in sentence.chunks:
                for morph in chunk.morphs:
                    if morph.pos == '動詞':
                        cases = []
                        modi_chunks = []
                        for src in chunk.srcs:           # 動詞の係り元chunkから助詞を探す
                            case = [morph.surface for morph in sentence.chunks[src].morphs if morph.pos == '助詞']
                            if len(case) > 0:           # 助詞を含むchunkの場合は、助詞と項を取得
                                cases = cases + case
                                modi_chunks.append(''.join(morph.surface for morph in sentence.chunks[src].morphs if morph.pos != '記号'))
                        if len(cases) > 0:
                            cases = sorted(list(set(cases)))
                            line = '{}\t{}\t{}'.format(morph.base, ' '.join(cases), ' '.join(modi_chunks))

                            print(line, file=f)


