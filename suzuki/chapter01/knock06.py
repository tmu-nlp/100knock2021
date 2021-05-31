#“paraparaparadise”と”paragraph”に含まれる文字bi-gramの集合を，
#それぞれ, XとYとして求め，XとYの和集合，積集合，差集合を求めよ．さらに，’se’というbi-gramがXおよびYに含まれるかどうかを調べよ．

def ngram(target, n):
    return [ target[idx:idx+n] for idx in range(len(target) - n + 1) ]

Input1 = "paraparaparadise"
Input2 = "paragraph"

X = set(ngram(Input1, 2))
Y = set(ngram(Input2, 2))

print("X{}".format(X))
print("Y{}".format(Y))
print("和集合{}".format(X | Y))
print("積集合{}".format(X & Y))
print("差集合{}".format(X - Y))

print("X ... {}".format({'se'} <= X))
print("Y ... {}".format({'se'} <= Y))