""'''
45 .動詞を述語，動詞に係っている文節の助詞を格と考え，述語と格をタブ区切り形式で出力せよ．
以下の事項をUNIXコマンドを用いて確認せよ:
コーパス中で頻出する述語と格パターンの組み合わせ
「行う」「なる」「与える」という動詞の格パターン（コーパス中で出現頻度の高い順に並べよ）'''


from knock41 import load_chunk


if __name__ ==  '__main__':
    filepath = './data/ai.ja/ai.ja.txt.parsed'
    res = load_chunk(filepath)
    with open('./ans45.txt','w', encoding='utf-8') as f:
        for sentence in res:
            for chunk in sentence.chunks:
                for morph in chunk.morphs:
                    if morph.pos == '動詞':
                        cases = []
                        for src in chunk.srcs:
                            cases = cases + [morph.surface for morph in sentence.chunks[src].morphs if morph.pos == '助詞']
                        if len(cases) > 0:
                            cases = sorted(list(set(cases)))
                            line = '{}\t{}'.format(morph.base, ' '.join(cases))
                            print(line, file=f)
                        #break
'''
$wc ./ans45.txt -l  ->974
$cat ./ans45.txt | sort | uniq -c | sort -nr | head -n 10
$cat ./ans45.txt |grep {'行う','なる'} | sort |uniq -c | sort -nr | head -n 5

'''
