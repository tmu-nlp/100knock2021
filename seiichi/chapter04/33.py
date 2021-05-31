import pickle

with open("./mecab.bin", "rb") as f:
    m = pickle.load(f)

def tg(sent):
    return zip(sent, sent[1:], sent[2:])

def check(x, y, z):
    return x['pos'] == z['pos'] == '名詞' and y['base'] == 'の'

ret = {''.join([x['surface'] for x in tri_gram]) for sent in m for tri_gram in tg(sent) if check(*tri_gram)}

for _, p in zip(range(30), ret):
    print(p)
print(len(ret))
