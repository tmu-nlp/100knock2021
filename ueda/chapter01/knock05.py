def n_gram(sents, n):
    gram = []
    for i in range(len(sents)):
        if i + n > len(sents): #超えた場合
            return gram
        gram.append(sents[i:i+n])


sent = "I am an NLPer"

print(n_gram(sent.split(), 2))
print(n_gram(sent, 2))
