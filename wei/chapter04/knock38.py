""'''
単語の出現頻度のヒストグラム（横軸に出現頻度，
縦軸に出現頻度をとる単語の種類数を棒グラフで表したもの）を描け．'''

if __name__ == '__main__':
    from knock30 import load_mecabf
    import matplotlib.pyplot as plt
    from collections import defaultdict


    mecabfile = './data/neko.txt.mecab'
    nekodata = load_mecabf(mecabfile)
    ans = defaultdict(int)
    for sentence in nekodata:
        for morphs in sentence:
            if morphs['pos'] != '記号':
                ans[morphs['base']] += 1
    ans = ans.values()

    plt.figure(figsize=(8,4))
    plt.hist(ans, bins=100)
    plt.xlabel('出現頻度')
    plt.ylabel('単語の種類数')
    plt.show()

