from knock30 import sentences
from collections import defaultdict
import matplotlib.pyplot as plt

ans=defaultdict(int)

for sentence in sentences:
    for mor in sentence:
        if mor['pos'] != '記号':
            ans[mor['base']] += 1
value=ans.values()

plt.figure(figsize=(8, 4))
plt.hist(value, bins=100)
plt.xlabel('出現頻度')
plt.ylabel('単語の種類数')
plt.show()