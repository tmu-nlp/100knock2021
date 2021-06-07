from knock40 import Morph
from knock41 import Chunks, chunk_list

if __name__ == '__main__':
    f = open('ai.ja.txt.parsed', 'r')
    w = open('answer47.txt', 'w')
    sentences = chunk_list(f)

    for sentence in sentences:
        for chank in sentence:
            for morph in chank.morphs:
                if morph.pos == '動詞': #最左でありかつ動詞
                    girls = set() #係り元の助詞のセット
                    koo = set() #係り元の項のセット
                    for src in chank.srcs: #動詞の係り元探索
                        souce_chank = sentence[src]
                        for morph2 in souce_chank.morphs:
                            if morph2.pos == '助詞':
                                predicate = ''
                                if souce_chank.morphs[0].pos == '名詞' and souce_chank.morphs[0].pos1 == 'サ変接続' and souce_chank.morphs[1].surface == 'を':
                                    predicate = souce_chank.morphs[0].surface + souce_chank.morphs[1].surface + morph.base #目的の述語の場合のみ len(predicate) > 0
                                    koo_dis = souce_chank.connect() #kooで述語に含まれるヲ格がかぶるので最後に消す用
                                koo.add(souce_chank.connect())
                                girls.add(morph2.base) #係り元に助詞があれば追加
                    if len(girls) > 0 and len(predicate) > 0:
                        koo.discard(koo_dis)
                        girls = sorted(list(girls))
                        koo = sorted(list(koo))
                        w.write(predicate + '\t' +  ' '.join(girls) + '\t' + ' '.join(koo) +  '\n')
                    break #最左の動詞のみを処理するためbreak

    w.close()
    f.close()

'''
出力
cat ./answer47.txt | cut -f 1,2 | sort | uniq -c | sort -nr | head -n 10

2 研究をする を
1 弾圧を併せ持つ     を
1 学習を繰り返す     を
1 注目を集める       から は を
1 進化を見せる       て において は を
1 反乱を起こす       て に対して を
1 追及を受ける       て で と とともに は を
1 禁止を求める       が に は を
1 研究を続ける       が て を
1 研究を進める       て を
'''