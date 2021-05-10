with open('popular-names.txt') as f:
    lines = [line.replace('\n', '').split('\t') for line in f.readlines()]
    lines = sorted(lines, reverse=True, key=lambda x: int(x[2]))
    
lines = ['\t'.join(line) for line in lines]
print('\n'.join(lines))

#sort -rnk 3 popular-names.txt