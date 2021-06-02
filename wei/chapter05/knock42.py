'''42. 係り元と係り先の文節の表示
係り元の文節と係り先の文節のテキストをタブ区切り形式ですべて抽出せよ．
ただし，句読点などの記号は出力しないようにせよ．'''

from knock40 import Morph
from knock41 import Chunk, parse_cabocha


if __name__ == '__main__':
    with open('./data/ai.ja/ai.ja.txt.parsed','r',encoding='utf-8') as f:
        blocks = f.read().split('EOS\n')
    blocks = list(filter(lambda x: x != '', blocks))
    blocks = [parse_cabocha(block) for block in blocks]


    for b in blocks:
        for m in b:
            if int(m.dst) > -1:
                print(''.join([mo.surface if mo.pos != '記号' else '' for mo in m.morphs]),
                      ''.join([mo.surface if mo.pos != '記号' else '' for mo in b[int(m.dst)].morphs]), sep = '\t')
