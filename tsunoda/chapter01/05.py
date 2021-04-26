
def ngram(n, lst):
  # ex.
  # [str[i:] for i in range(2)] -> ['I am an NLPer', ' am an NLPer']
  # zip(*[str[i:] for i in range(2)]) -> zip('I am an NLPer', ' am an NLPer')
  return list(zip(*[lst[i:] for i in range(n)]))

str = 'I am an NLPer'
words_bi_gram = ngram(2, str.split())
chars_bi_gram = ngram(2, str)

print('単語bi-gram:', words_bi_gram)
print('文字bi-gram:', chars_bi_gram)