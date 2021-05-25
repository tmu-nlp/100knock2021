def cipher(sents):
    sents = [chr(219-ord(words)) if words.islower() else words for words in sents]
    return ''.join(sents)

sent = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
sent = cipher(sent)
print(sent)
print(cipher(sent))