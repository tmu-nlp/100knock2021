from knock35 import word_count
import matplotlib.pyplot as plt

with open('neko.txt.mecab') as f:
    words = word_count(f.readlines())

keys = [word[0] for word in words[:10]]
vals = [word[1] for word in words[:10]]

plt.figure()
plt.bar(keys, vals)
plt.savefig('knock36.png')
