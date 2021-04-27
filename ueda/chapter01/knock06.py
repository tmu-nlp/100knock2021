def n_gram(sents, n):
    gram = []
    for i in range(len(sents)):
        if i + n > len(sents): #超えた場合
            return gram
        gram.append(sents[i:i+n])

word_1, word_2 = "paraparaparadise",  "paragraph"
X,Y=n_gram(word_1, 2), n_gram(word_2, 2)
print("和集合: {}\n積集合： {}\n差集合: {}\n".format(set(X)|set(Y),set(X)&set(Y),set(X)-set(Y)))
print("Xに\"se\"が含まれる: {}\nYに\"se\"が含まれる: {}\n".format(str("se" in X), str("se" in Y)))