from janome.tokenizer import Tokenizer
t = Tokenizer(wakati=True)

def n_gram_char(sentence : str, n : int):
	return [sentence[i:i+n] for i in range(len(sentence) - n + 1)]

def n_gram_words(words : list, n : int):
	res =[]
	tmp = [words[i:i+n] for i in range(len(words) - n + 1)]
	for bigram in tmp:
		ans = ''
		for word in bigram:
			ans+=word
		res.append(ans)
	return res

sentence = 'I am an NLPer'

char_bi_gram = n_gram_char(sentence,2)
words_bi_gram =n_gram_words(list(t.tokenize(sentence)), 2)
print(char_bi_gram,words_bi_gram)
