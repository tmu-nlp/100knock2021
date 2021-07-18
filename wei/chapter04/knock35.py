""'''
35. 文章中に出現する単語とその出現頻度を求め，出現頻度の高い順に並べよ'''


from collections import defaultdict
from knock30 import load_mecabf


def sort_cnts(sentences):
    ans = defaultdict(int)
    for sentence in sentences:
        for morphs in sentence:
            if morphs['pos'] != '記号':
                ans[morphs['base']] += 1
    ans = sorted(ans.items(), key= lambda x: x[1], reverse=True)   # 出現頻度の高い順に並べる
    return ans


if __name__ == '__main__':
    mecabfile = './data/neko.txt.mecab'
    nekodata = load_mecabf(mecabfile)
    ans = sort_cnts(nekodata)
    for v in ans[:10]:
        print(v)


'''
('の', 9194)
('て', 6848)
('は', 6420)
('に', 6243)
('を', 6071)
('だ', 5975)
('と', 5508)
('が', 5337)
('た', 4267)
('する', 3657)
'''