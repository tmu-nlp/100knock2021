""'''
37. 「猫」とよく共起する（共起頻度が高い）10語とその出現頻度を
グラフ（例えば棒グラフなど）で表示せよ．'''

if __name__ == '__main__':
    from collections import defaultdict
    from knock30 import load_mecabf
    import matplotlib.pyplot as plt
    import japanize_matplotlib


    mecabfile = './data/neko.txt.mecab'
    nekodata = load_mecabf(mecabfile)
    ans = defaultdict(int)
    for sentence in nekodata:
        if '猫' in [morphs['surface'] for morphs in sentence]:
            for morphs in sentence:
                if morphs['pos'] != '記号':
                    ans[morphs['base']] += 1
    del ans['猫']

    ans = sorted(ans.items(), key=lambda x:x[1], reverse=True)
    words = [i[0] for i in ans[:10]]
    cnts = [i[1] for i in ans[:10]]

    plt.figure(figsize=(8, 4))
    plt.bar(words, cnts)
    plt.show()


