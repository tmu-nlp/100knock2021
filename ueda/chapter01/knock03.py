import re
sent = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
print([len(re.sub(r"[,.]", "", sent)) for sent in sent.split()])