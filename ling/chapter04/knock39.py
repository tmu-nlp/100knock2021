from knock30 import sentences
from collections import defaultdict
import matplotlib.pyplot as plt
ans=defaultdict(int)

for sentence in sentences:
    for mor in sentence:
        if mor['pos']!='記号':
            ans[mor['base']]+=1

ans=sorted(ans.items(),key=lambda x:x[1],reverse=True)

ranks = [r + 1 for r in range(len(ans))]
values = [a[1] for a in ans]
plt.figure(figsize=(8, 4))
plt.scatter(ranks, values)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('出現頻度順位')
plt.ylabel('出現頻度')
plt.show()