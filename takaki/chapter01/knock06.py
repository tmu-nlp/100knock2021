def ngram(l, n):
    return list(zip(*[l[i:] for i in range(n)]))

x = set(ngram("paraparaparadise", 2))
y = set(ngram("paragraph"       , 2))

print(f"和集合: {x | y}")
print(f"積集合: {x & y}")
print(f"差集合: {x - y}")
print('se <= x:', {('s', 'e')} <= x)
print('se <= y:', {('s', 'e')} <= y)
