ls = set()
with open('popular-names.txt') as f:
    for line in f:
        words = line.split('\t')
        ls.add(words[0])

print('\n'.join(sorted(ls)))

#cut -f 1 popular-names.txt | sort | uniq