import matplotlib.pyplot as plt
import numpy as np
from knock35 import count
from collections import defaultdict
cnt = []
for w in count:
    cnt.append(w[1])

plt.figure(figsize=(8, 4))
plt.hist(cnt, bins=50)
plt.xlabel('出現頻度')
plt.ylabel('単語の種類数')
plt.show()