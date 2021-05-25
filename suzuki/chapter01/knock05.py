#与えられたシーケンス（文字列やリストなど）からn-gramを作る関数を作成せよ．
#この関数を用い，”I am an NLPer”という文から単語bi-gram，文字bi-gramを得よ．

def ngram(target, n):
    return [ target[idx:idx+n] for idx in range(len(target) - n + 1) ]

Input = "I am an NLPer"

print(ngram(Input, 2))

words = Input.split(" ")
print(ngram(words, 2))