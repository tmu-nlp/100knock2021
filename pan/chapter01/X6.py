##Let the sets of letter bi-grams from the words “paraparaparadise” and “paragraph” $X$ and $Y$, respectively. Obtain the union, intersection, difference of the two sets. In addition, check whether the bigram “se” is included in the sets $X$ and $Y$
def ngram(sentence,n):
    return [sentence[x:x+n] for x in range(len(sentence)- n + 1)]
s1 = 'paraparaparadise'
s2 = 'paragraph'
a = set(ngram(s1,2))
b = set(ngram(s2,2))
print('X', a)
print('Y', b)
print('Union', X | Y)
print('Inter', X & Y)
print('Diff', X - Y)
print('Is it in the X:','se' in X)
print('Is it in the Y:','se' in Y)

#X {'pa', 'ap', 'ad', 'ra', 'ar', 'is', 'se', 'di'}
#Y {'pa', 'gr', 'ap', 'ra', 'ar', 'ph', 'ag'}
#Union {'pa', 'gr', 'ap', 'ad', 'ra', 'ar', 'is', 'se', 'ph', 'di', 'ag'}
#Inter {'ar', 'pa', 'ap', 'ra'}
#Diff {'di', 'ad', 'is', 'se'}
#Is it in the X: True
#Is it in the Y: False
