""'''
34. 名詞の連接（連続して出現する名詞）を最長一致で抽出せよ．'''
from knock30 import load_mecabf

def longest_nouns(sentences):
    ans = set()
    for sentence in sentences:
        nouns = ''
        num = 0
        for morphs in sentence:
            if morphs['pos'] == '名詞':
                nouns = ''.join([nouns, morphs['surface']])
                num += 1
            elif num >= 2:  # 名詞以外の場合、ここまでの連結数が2以上の場合は出力し、nounsとnumを初期化
                ans.add(nouns)
                nouns = ''
                num = 0
            else:
                nouns = ''
                num = 0

        if num >= 2:
            ans.add(nouns)

    return ans


if __name__ == '__main__':
    mecabfile = './data/neko.txt.mecab'
    nekodata = load_mecabf(mecabfile)
    nouns = longest_nouns(nekodata)
    print(f'最長一致の連接名詞の種類:{len(nouns)}')
    print('---samples---')
    for k, v in enumerate(list(nouns)[:10]):
        print(k, '\t', v)

'''
最長一致の連接名詞の種類:4457
---samples---
0 	 学問上
1 	 変出鱈目
2 	 理想的
3 	 近所合壁有名
4 	 時下秋冷
5 	 一二滴眼尻
6 	 柔術使
7 	 なんざ
8 	 ただ顔
9 	 鮑貝'''