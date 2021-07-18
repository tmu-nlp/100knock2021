""'''
39. Zipfの法則
単語の出現頻度順位を横軸，その出現頻度を縦軸として，
両対数グラフをプロットせよ．'''

if __name__ == '__main__':
    import math
    from knock30 import load_mecabf
    from knock35 import sort_cnts
    import matplotlib.pyplot as plt
    import japanize_matplotlib


    mecabfile = './data/neko.txt.mecab'
    nekodata = load_mecabf(mecabfile)
    ans = sort_cnts(nekodata)


    ranks = [r + 1 for r in range(len(ans))]
    cnts = [i[1] for i in ans]
    plt.figure(figsize=(8, 4))
    plt.scatter(ranks, cnts)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('出現頻度順位')
    plt.ylabel('出現頻度')
    plt.show()


