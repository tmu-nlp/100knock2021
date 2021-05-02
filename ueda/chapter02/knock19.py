from collections import defaultdict
dicts = defaultdict(lambda: 0)
with open('c:\Git\popular-names.txt') as f:
    for line in f:
        dicts[line.split('\t')[0]] += 1
for foo, bar in sorted(dicts.items(), key=lambda x:x[1], reverse=True):
    print("%s %r" % (foo, bar))