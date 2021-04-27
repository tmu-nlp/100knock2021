with open('out_py/col1.txt') as f1, open('out_py/col2.txt') as f2:
    with open('out_py/13.txt', 'w') as fout:
        for a, b in zip(f1.readlines(), f2.readlines()):
            fout.write(f'{a.strip()}\t{b.strip()}\n')
