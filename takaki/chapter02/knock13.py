x = []
with open('tmp/col1.py.txt') as f1, open('tmp/col2.py.txt') as f2:
    for a, b in zip(f1.readlines(), f2.readlines()):
        x.append(f"{a.strip()}\t{b.strip()}\n")
with open('tmp/knock13.py.txt', 'w') as f:
    f.writelines(x)
