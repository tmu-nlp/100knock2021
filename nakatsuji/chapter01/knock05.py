def n_gram(n, S):
    return [S[idx:idx + n] for idx in range(len(S) - n + 1)]

S = "I am an NLPer"
print(n_gram(2, S))
print(n_gram(2, S.split(' ')))