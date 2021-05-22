import matplotlib.pyplot as plt
from knock35 import count
top10 = count[:10]
left = []
height = []
for w in top10:
    print(w[0])
    left.append(w[0])
    height.append(w[1])
plt.figure()
plt.bar(left, height)
plt.show()