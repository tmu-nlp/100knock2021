def n_gram(n, S):
    return [S[idx:idx + n] for idx in range(len(S) - n + 1)]

S = "I am an NLPer"
for i in range(1, 4):
    print(n_gram(i, S))
    print(n_gram(i, S.split(' ')))