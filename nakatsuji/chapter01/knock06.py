def n_gram(target, n):
    return [target[idx:idx + n] for idx in range(len(target) - n + 1)]

S1 = 'paraparaparadise'
S2 = 'paragraph'
X = n_gram(S1, 2)
Y = n_gram(S2, 2)
print("和集合：{}\n積集合：{}\n差集合：{}".format(set(X) | set(Y), set(X) & set(Y), set(X) - set(Y)))
print('se' in (set(X) & set(Y)))