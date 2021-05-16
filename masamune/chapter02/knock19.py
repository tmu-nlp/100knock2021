from collections import defaultdict

with open('popular-names.txt') as f:
    lines = f.readlines()
    words = [line.replace('\n', '').split('\t')[0] for line in lines]

chara_cnt = defaultdict(lambda: 0)
for word in words:
    chara_cnt[word] += 1

chara_cnt = sorted(chara_cnt.items(), reverse=True, key = lambda x: x[1])
print(*[f"{word} {cnt}" for word, cnt in chara_cnt], sep='\n')

#cut -f 1 popular-names.txt | sort | uniq -c | sort -rn