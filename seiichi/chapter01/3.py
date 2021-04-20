import re
sent = 'Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics.'
print(''.join([str(len(re.sub(r'[^a-zA-Z]', '', s))) for s in sent.split()]))
