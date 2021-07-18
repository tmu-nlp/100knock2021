""'''
36. 出現頻度が高い10語とその出現頻度をグラフ（例えば棒グラフなど）で表示せよ．'''

if __name__ == '__main__':
    from knock30 import load_mecabf
    from knock35 import sort_cnts
    import matplotlib.pyplot as plt
    import japanize_matplotlib


    mecabfile = './data/neko.txt.mecab'
    nekodata = load_mecabf(mecabfile)
    ans = sort_cnts(nekodata)
    words = [i[0] for i in ans[:10]]
    cnts = [i[1] for i in ans[:10]]

    plt.figure(figsize=(8,4))
    plt.bar(words, cnts)
    plt.show()
