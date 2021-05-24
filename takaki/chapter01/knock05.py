def ngram(l, n):
    return list(zip(*[l[i:] for i in range(n)]))

print(ngram("I am an NLPer".split(), 2))
print(ngram("I am an NLPer", 2))
