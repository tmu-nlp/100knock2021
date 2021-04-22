import random

from janome.tokenizer import Tokenizer
t = Tokenizer(wakati=True)


sentence = "I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind ."

def Typoglycemia(sentence : str):
	words = list(t.tokenize(sentence))
	if len(words) < 4:
		return
	tmp = ''
	for i in range(1,len(words) - 1):
		tmp += words[i]
	shuffled = ''.join(random.sample(tmp, len(tmp)))
	return words[0] + shuffled + words[len(words) - 1]

	# print(shuffled)

print(Typoglycemia(sentence))