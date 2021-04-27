import random
def shuffle(word):
    if len(word) <= 4:
        return word
    else:
        s, t = word[0], word[1]
        middle = random.sample(list(word[1:-1]), len(word[1:-1]))
        return ''.join([s] + middle + [t])

text = "I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind ."
ans = [shuffle(word) for word in text.split()]
print(' '.join(ans))