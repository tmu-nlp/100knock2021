""'''
33.2つの名詞が「の」で連結されている名詞句を抽出せよ．
'''

from knock30 import load_mecabf


def find_nouns(sentences):
    results = set()
    for sentence in sentences:
        for i in range(1, len(sentence) - 1):
            if sentence[i-1]['pos'] == '名詞' and sentence[i]['surface'] == 'の' and sentence[i+1]['pos'] == '名詞':
                results.add(sentence[i-1]['surface'] + sentence[i]['surface'] + sentence[i+1]['surface'])
    return results


if __name__ == '__main__':

    mecabfile = './data/neko.txt.mecab'
    nekodata = load_mecabf(mecabfile)
    nouns = find_nouns(nekodata)
    print(f'「名詞＋の＋名詞」の種類:{len(nouns)}')
    print('---samples---')
    for k, v in enumerate(list(nouns)[:10]):
        print(k, '\t', v)

'''
「名詞＋の＋名詞」の種類:4924
---samples---
0 	 近所の女学校
1 	 独身の僕
2 	 水車の勢
3 	 人の運命
4 	 熊の画
5 	 下駄の音
6 	 トチメンボーの復讐
7 	 今の世
8 	 あちらの方角
9 	 野暮の方
'''