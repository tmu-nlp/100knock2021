from collections import Counter

tar = open('./dat/popular-names.txt').readlines()
pref_counter = Counter(line.split('\t')[0] for line in tar)


with open('./out_py/19.txt', 'w') as f:
    for k, v in pref_counter.most_common():
        f.write(f'{v} {k}\n')
