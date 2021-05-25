import random

def typoglycemia(sents):
    sents = sents.split()
    result = [words[0]+''.join(random.sample(words[1:-1], len(words[1:-1])))+words[-1] if len(words) >= 4 else words for words in sents]
    return ' '.join(result)

sent = "I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind ."
print(typoglycemia(sent))