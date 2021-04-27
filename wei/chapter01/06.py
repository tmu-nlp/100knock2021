# 06. 集合
# "paraparaparadise"と"paragraph"に含まれる文字bi-gramの集合を，それぞれ, XとYとして求め，XとYの和集合，積集合，差集合を求めよ．
# さらに，'se'というbi-gramがXおよびYに含まれるかどうかを調べよ．


# 文字bi-gramの集合を作る関数を作成
def n_gram(wlist, n):
    n_list = list(zip(*[wlist[i:] for i in range(n)]))
    list_set = set(n_list)
    return list_set


if __name__ == '__main__':
    str_a = list("paraparaparadise")
    str_b = list("paragraph")

# 　xとyを求める
x = n_gram(str_a, 2)
y = n_gram(str_b, 2)
print
'x = ', x
print
'y = ', y

# 　xとyの和集合、積集合、差集合を求める
print
'x + y = ', x | y
print
'x & y = ', x & y
print
'x - y = ', x - y
print
'y - x = ', y - x


# 'se'というbi-gramがXおよびYに含まれるかどうかを調べ
def is_in_bigram(str_set):
    if ('s', 'e') in str_set:
        result = bool(1)
    else:
        result = bool(0)
    return result


print
'\'se\' in x = ', is_in_bigram(x)
print
'\'se\' in y = ', is_in_bigram(y)