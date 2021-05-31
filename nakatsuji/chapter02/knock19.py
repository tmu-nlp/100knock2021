from collections import defaultdict
with open("popular-names.txt") as f:
    lines = f.readlines()
    cnt_num = defaultdict(lambda: 0)
    for line in lines:
        cnt_num[line.split('\t')[0]] += 1
    cnt_lis = list(sorted(cnt_num.items(), key = lambda x:x[1] ,reverse=True))

    with open('py/knock19.txt', 'w') as f1:
        for cnt in cnt_lis:
            f1.write(f'{cnt[1]} ' + cnt[0]+'\n')