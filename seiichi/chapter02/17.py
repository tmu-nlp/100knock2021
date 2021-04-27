with open('./out_py/17.txt', 'w') as f:
    f.write('\n'.join(sorted(list(set([line.split('\t')[0] for line in open('./dat/popular-names.txt', 'r').readlines()])))))

