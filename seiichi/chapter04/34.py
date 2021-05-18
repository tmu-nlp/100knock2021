import pickle

with open("./mecab.bin", "rb") as f:
    m = pickle.load(f)

def ln(sent):
    l = []
    tmp = []
    for d in sent:
        if d['pos'] == '名詞':
            tmp.append(d['surface'])
        else:
            if len(tmp) > 1:
                l.append(tmp)
            tmp = []
    return l

ret = [''.join(n) for sent in m for n in ln(sent)]

for _, c in zip(range(30), ret):
    print(c)
print(len(ret))

