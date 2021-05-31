from knock30 import parse2_mecab
import matplotlib.pyplot as plt
from collections import defaultdict

with open('neko.txt.mecab') as f:
    lines = parse2_mecab(f.readlines())

print(lines)

words = defaultdict(int)

for line in lines:
    flag = False
    tmp  = defaultdict(int)
    for morph in line:
        if morph['surface'] == '猫':
            flag = True
        tmp[morph['base']] += 1
    if flag:
        for key, val in tmp.items():
            if key != '猫':
                words[key] += val

words = sorted(words.items(), key=lambda x:x[1], reverse=True)
keys = [word[0] for word in words[:10]]
vals = [word[1] for word in words[:10]]

plt.figure()
plt.bar(keys, vals)
plt.savefig('knock37.png')
