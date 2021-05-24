from knock30 import sentences
from collections import defaultdict
import matplotlib as mpl
import matplotlib.pyplot as plt

ans=defaultdict(int)

for sentence in sentences:
    if '猫' in [mor['surface'] for mor in sentence]:
        for mor in sentence:
            if mor['pos']!='記号':
                ans[mor['base']]+=1

del ans['猫']

ans=sorted(ans.items(),key=lambda x:x[1],reverse=True)

key=[a[0] for a in ans[:10]]
value=[a[1] for a in ans[:10]]

plt.figure(figsize=(8,6))
plt.bar(key,value)
plt.show()