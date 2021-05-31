from knock35 import word_count
import matplotlib.pyplot as plt

with open('neko.txt.mecab') as f:
    words = word_count(f.readlines())

rank = [r+1 for r in range(len(words))]
vals = [word[1] for word in words]

plt.figure()
plt.scatter(rank, vals)
plt.xscale('log')
plt.yscale('log')
plt.savefig('knock39.png')
