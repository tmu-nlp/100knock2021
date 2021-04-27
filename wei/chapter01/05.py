# 05. n-gram(单词预测模型)
# 与えられたシーケンス（文字列やリストなど）からn-gramを作る関数を作成せよ．この関数を用い，
# "I am an NLPer"という文から単語bi-gram，文字bi-gramを得よ．

# 定义一个n-gram函数
def n_gram(wlist, n):
    n_wlist = []
    for i in range(n):
        n_wlist.append(wlist[i:])
    return zip(*n_wlist)


if __name__ == '__main__':
    raw_t = 'I am an NLPer'
    w_list = raw_t.strip().split()
    c_list = list(raw_t)

    print
    n_gram(w_list, 2)
    print
    n_gram(c_list, 2)