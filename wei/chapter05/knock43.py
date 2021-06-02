'''43. 名詞を含む文節が動詞を含む文節に係るものを抽出
名詞を含む文節が，動詞を含む文節に係るとき，これらをタブ区切り形式で抽出せよ．
ただし，句読点などの記号は出力しないようにせよ．'''

from knock41 import Chunk, parse_cabocha


if __name__ == '__main__':
    with open('./data/ai.ja/ai.ja.txt.parsed','r',encoding='utf-8') as f:
        blocks = f.read().split('EOS\n')
    blocks = list(filter(lambda x: x != '', blocks))
    blocks = [parse_cabocha(block) for block in blocks]


    for b in blocks:
        for m in b:
            if int(m.dst) > -1:
                pre_text = ''.join([mo.surface if mo.pos != '記号' else '' for mo in m.morphs])
                pre_pos = [mo.pos for mo in m.morphs]
                post_text = ''.join([mo.surface if mo.pos != '記号' else '' for mo in b[int(m.dst)].morphs])
                post_pos = [mo.pos for mo in b[int(m.dst)].morphs]
                if '名詞' in pre_pos and '動詞' in post_pos:
                    print(pre_text, post_text, sep = '\t')