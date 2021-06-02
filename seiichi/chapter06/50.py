import random
from collections import Counter
from tabulate import tabulate

random.seed(0)

def wt(data, path):
    tmp = "\n".join(["\t".join(l) for l in data])
    with open(path, "w") as f:
        f.write(tmp)

col_names = ['ID','TITLE', 'URL', 'PUBLISHER' 'CATEGORY','STORY', 'HOSTNAME', 'TIMESTAMP']
with open("./data/NewsAggregatorDataset/newsCorpora.csv") as f:
    tmp = f.readlines()
data = [line.split('\t') for line in tmp]

publishers = {'Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail',}
data_extracted = [l for l in data if l[3] in publishers]

data_all = [[l[4], l[1]] for l in data_extracted]
random.shuffle(data_all)
tr, va = int(len(data_all) * 0.8), int(len(data_all) * 0.9)
train, valid, test = data_all[:tr], data_all[tr:va], data_all[va:]

print("all: {}, train: {}, valid: {}, test: {}".format(len(data_all), len(train), len(valid), len(test)))

wt(train, "data/train.txt")
wt(valid, "data/valid.txt")
wt(test, "data/test.txt")

categories = ['b', 't', 'e', 'm']
category_names = ['business', 'science and technology', 'entertainment', 'health']
table = [
    [name] + [freqs[cat] for cat in categories]
    for name, freqs in [
        ('train', Counter([cat for cat, _ in train])),
        ('valid', Counter([cat for cat, _ in valid])),
        ('test', Counter([cat for cat, _ in test])),
    ]
]
print(tabulate(table, headers=category_names))
