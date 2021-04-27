with open('out_py/col1.txt', 'w') as f, open('out_py/col2.txt', 'w') as f2:
    for line in open('dat/popular-names.txt').readlines():
        cols = line.split('\t')
        f.write(cols[0]+'\n')
        f2.write(cols[1]+'\n')
