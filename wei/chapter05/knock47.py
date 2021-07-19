""'''
47. 機能動詞構文のマイニング
動詞のヲ格にサ変接続名詞が入っている場合のみに着目したい。
「サ変接続名詞＋を＋動詞の基本形」
述語に係る助詞(文節)が複数ある場合、全ての助詞をスペース区切りで辞書順に並べ、
全ての項をスペース区切りで並べる'''

from knock41 import load_chunk


if __name__ ==  '__main__':
    filepath = './data/ai.ja/ai.ja.txt.parsed'
    res = load_chunk(filepath)
    with open('./ans47.txt','w', encoding='utf-8') as f:
        for sentence in res:
            for chunk in sentence.chunks:
                for morph in chunk.morphs:
                    if morph.pos == '動詞':
                        for i, src in enumerate(chunk.srcs):
                            if len(sentence.chunks[src].morphs) == 2 and sentence.chunks[src].morphs[0].pos1 == 'サ変接続' and sentence.chunks[src].morphs[1].surface == 'を':
                                predict = ''.join([sentence.chunks[src].morphs[0].surface, sentence.chunks[src].morphs[1].surface, morph.base])
                                cases = []
                                modi_chunks = []
                                for src_r in chunk.srcs[:i] + chunk.srcs[i+1:]:           # 動詞の係り元chunkから助詞を探す
                                    case = [morph.surface for morph in sentence.chunks[src_r].morphs if morph.pos == '助詞']
                                    if len(case) > 0:           # 助詞を含むchunkの場合は、助詞と項を取得
                                        cases = cases + case
                                        modi_chunks.append(''.join(morph.surface for morph in sentence.chunks[src_r].morphs if morph.pos != '記号'))
                                if len(cases) > 0:
                                    cases = sorted(list(set(cases)))
                                    line = '{}\t{}\t{}'.format(predict, ' '.join(cases), ' '.join(modi_chunks))

                                    print(line, file=f)

'''
cat ./ans47.txt | cut -f 1 | sort | uniq -c | sort -nr | head -n 10
cat ./ans47.txt | cut -f 1,2 | sort | uniq -c | sort -nr | head -n 10
'''