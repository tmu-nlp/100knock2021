import string
s = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
print([len(word) for word in s.translate(str.maketrans('', '', string.punctuation)).split()])
