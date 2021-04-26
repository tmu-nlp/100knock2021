def ngram(target, n):
  return [target[idx:idx + n] for idx in range(len(target) - n + 1)]
s1, s2 = 'paraparaparadise', 'paragraph'
x, y = set(ngram(s1, 2)), set(ngram(s2, 2))
print(x | y)
print(x & y)
print(x - y)
print('se' in x, 'se' in y)
