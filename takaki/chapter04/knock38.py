from knock35 import word_count
import matplotlib.pyplot as plt

with open('neko.txt.mecab') as f:
    words = word_count(f.readlines())

vals = [word[1] for word in words]

plt.figure()
plt.hist(vals)
plt.savefig('knock38.png')
