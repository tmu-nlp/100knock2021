c = {}
with open('popular-names.txt') as f:
    for line in f.readlines():
        x = line.split()[0]
        c[x] = True
print(len(c.keys()))
