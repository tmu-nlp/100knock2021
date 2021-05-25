# 05. n-gram(单词预测模型)
# 与えられたシーケンス（文字列やリストなど）からn-gramを作る関数を作成せよ．この関数を用い，
# "I am an NLPer"という文から単語bi-gram，文字bi-gramを得よ．

# 定义一个n-gram函数
def n_gram(wlist, n):
    n_wlist = []
    for i in range(n, len(wlist)+1):
        n_wlist.append('|'.join(wlist[i-n:i]))  # ''.join(list)：列表转字符串，''中为字符串分隔符，将list转为字符串
        # print(n_wlist)

    return n_wlist

if __name__ == '__main__':
    raw_t = 'I am an NLPer'
    w_list = raw_t.strip().split()

    c_list = list(raw_t)

    print('Word level bi-gram:', n_gram(w_list, 2))
    print('Character level bi-gram:', n_gram(raw_t, 2))

