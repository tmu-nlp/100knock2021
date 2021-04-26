def ngram(target, n):
  return [target[idx:idx + n] for idx in range(len(target) - n + 1)]
sent = 'I am an NLPer'
print(ngram(sent, 1))
print(ngram(sent, 2))
print(ngram(sent.split(), 1))
print(ngram(sent.split(), 2))
