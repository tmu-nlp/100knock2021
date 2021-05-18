with open('py/knock13_py.txt', 'w') as f:
    with open('py/col1.txt') as m1, open('py/col2.txt') as m2:
        for w1, w2 in zip(m1.readlines(), m2.readlines()):
            line =w1.strip()+'\t'+w2.strip()
            f.write(line+'\n')