def n_gram_char(sentence : str, n : int):
	return [sentence[i:i+n] for i in range(len(sentence) - n + 1)]

str1 = 'paraparaparadise'
str2 = 'paragraph'
target = set()
target.add('se')

X = set(n_gram_char(str1, 2))
Y = set(n_gram_char(str2, 2))
Union = X | Y
Inter = X & Y
Diff = X - Y

print('X',X)
print('Y',Y)
print('Union',Union)
print('Inter',Inter)
print('Diff', Diff)

print('target is in X?', target <= X)
print('target is in Y?', target <= Y)
