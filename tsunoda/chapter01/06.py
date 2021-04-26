def ngram(n, lst):
  # ex.
  # [str[i:] for i in range(2)] -> ['I am an NLPer', ' am an NLPer']
  # zip(*[str[i:] for i in range(2)]) -> zip('I am an NLPer', ' am an NLPer')
  return list(zip(*[lst[i:] for i in range(n)]))
str1 = 'paraparaparadise'
str2 = 'paragraph'
X = set(ngram(2, str1))
Y = set(ngram(2, str2))
union = X | Y
intersection = X & Y
difference = X - Y

print('X:', X)
print('Y:', Y)
print('和集合:', union)
print('積集合:', intersection)
print('差集合:', difference)
print('Xにseが含まれるか:', {('s', 'e')} <= X)
print('Yにseが含まれるか:', {('s', 'e')} <= Y)