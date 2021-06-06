import random
import collections

with open(r'C:\Git\newsCorpora.csv', encoding="utf-8") as f:
    corpora =[]
    for line in f:
        line = line.split('\t')
        if line[3] not in ["Reuters", "Huffington Post", "Businessweek", "Contactmusic.com", "Daily Mail"]:
            continue
        corpora.append(line)
    random.shuffle(corpora)
    with open(r'C:\Git\train.txt', 'w', encoding="utf-8") as train, open(r'C:\Git\valid.txt', 'w',encoding="utf-8") as valid, open(r'C:\Git\test.txt', 'w',encoding="utf-8") as test:
        train.writelines(a[4]+'\t'+a[1]+'\n' for a in corpora[:int(len(corpora)*0.8)])
        valid.writelines(a[4]+'\t'+a[1]+'\n' for a in corpora[int(len(corpora)*0.8)+1:int(len(corpora)*0.9)])
        test.writelines(a[4]+'\t'+a[1]+'\n' for a in corpora[int(len(corpora)*0.9)+1:])
        print(collections.Counter([a[4] for a in corpora[:int(len(corpora)*0.8)]]))
        print(collections.Counter([a[4] for a in corpora[int(len(corpora)*0.8)+1:int(len(corpora)*0.9)]]))
        print(collections.Counter([a[4] for a in corpora[int(len(corpora)*0.9)+1:]]))

            

    

        