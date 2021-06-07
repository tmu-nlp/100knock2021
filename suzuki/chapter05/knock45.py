from knock40 import Morph
from knock41 import Chunks, chunk_list

if __name__ == '__main__':
    f = open('ai.ja.txt.parsed', 'r')
    w = open('answer45.txt', 'w')
    sentences = chunk_list(f)

    for sentence in sentences:
        for chank in sentence:
            for morph in chank.morphs:
                if morph.pos == '動詞': #最左でありかつ動詞
                    predicate = morph.base #述語は動詞の基本形
                    girls = set() #係り元の助詞のセット
                    for src in chank.srcs: #動詞の係り元探索
                        souce_chank = sentence[src]
                        for morph2 in souce_chank.morphs:
                            if morph2.pos == '助詞':
                                girls.add(morph2.base) #係り元に助詞があれば追加
                    girls = sorted(list(girls))
                    if len(girls) > 0:
                        w.write(predicate + '\t' +  ' '.join(girls) + '\n')
                    break #最左の動詞のみを処理するためbreak

    w.close()
    f.close()

'''
コーパス中で頻出する述語と格パターンの組み合わせ
cat ./answer45.txt | sort | uniq -c | sort -nr | head -n 10
49 する       を
19 する       が
15 する       に
15 する       と
12 する       は を
11 れる       と
10 する       に を
9 する       で を
9 よる       に
8 する       が に

「行う」「なる」「与える」という動詞の格パターン（コーパス中で出現頻度の高い順に並べよ）
cat ./answer45.txt | grep '行う' | sort | uniq -c | sort -nr | head -n 5
8 行う       を
1 行う       まで を
1 行う       から
1 行う       に により を
1 行う       に まで を

cat ./answer45.txt | grep 'なる' | sort | uniq -c | sort -nr | head -n 5
4 なる       に は
3 なる       が と
2 なる       に
2 なる       と
1 無くなる   は

cat ./answer45.txt | grep '与える' | sort | uniq -c | sort -nr | head -n 5
1 与える     が など に
1 与える     に は を
1 与える     が に
'''
