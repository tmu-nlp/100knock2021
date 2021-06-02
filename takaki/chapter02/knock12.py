col1, col2 = [], []
with open('popular-names.txt') as f:
    for line in f.readlines():
        x = line.split()
        col1.append(x[0] + '\n')
        col2.append(x[1] + '\n')
with open('tmp/col1.py.txt', 'w') as f1, open('tmp/col2.py.txt', 'w') as f2:
    f1.writelines(col1)
    f2.writelines(col2)
