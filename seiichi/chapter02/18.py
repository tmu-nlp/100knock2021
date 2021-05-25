tar = open('./dat/popular-names.txt').readlines()
sorted_lines = sorted(tar, key=lambda line: float(line.split('\t')[2]), reverse=True)
with open('./out_py/18.txt', 'w') as f:
    f.write(''.join(sorted_lines))
