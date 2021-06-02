from collections import Counter

with open('popular-names.txt', 'r') as f:
    c = Counter([line.split()[0] for line in f.readlines()])
for x, y in c.most_common():
    print(f'{y}\t{x}')
