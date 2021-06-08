from knock40 import Morph
from knock41 import Chunks, chunk_list

if __name__ == '__main__':
    f = open('ai.ja.txt.parsed', 'r')
    w = open('answer46.txt', 'w')
    sentences = chunk_list(f)

    for sentence in sentences:
        for chank in sentence:
            for morph in chank.morphs:
                if morph.pos == '動詞': #最左でありかつ動詞
                    predicate = morph.base #述語は動詞の基本形
                    girls = set() #係り元の助詞のセット
                    koo = set() #係り元の項のセット
                    for src in chank.srcs: #動詞の係り元探索
                        souce_chank = sentence[src]
                        for morph2 in souce_chank.morphs:
                            if morph2.pos == '助詞':
                                girls.add(morph2.base) #係り元に助詞があれば追加
                                koo.add(souce_chank.connect())
                    girls = sorted(list(girls))
                    koo = sorted(list(koo))
                    if len(girls) > 0 and len(koo) > 0:
                        w.write(predicate + '\t' +  ' '.join(girls) + '\t' + ' '.join(koo) +  '\n')
                    break #最左の動詞のみを処理するためbreak

    w.close()
    f.close()

'''
出力
cat ./answer46.txt | head -n 10

用いる  を      道具を
する    て を   用いて 知能を
指す    を      一分野を
代わる  に を   人間に 知的行動を
行う    て に   コンピューターに 代わって
する    と も   研究分野とも
述べる  で に の は     佐藤理史は 次のように 解説で
する    で を   コンピュータ上で 知的能力を
する    を      推論判断を
する    を      画像データを

'''