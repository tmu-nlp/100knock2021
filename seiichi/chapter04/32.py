import pickle

with open("./mecab.bin", "rb") as f:
    m = pickle.load(f)

bov = {d['base'] for sent in m for d in sent if d['pos'] == '動詞'}
for _, v in zip(range(30), bov):
    print(v)
print(len(bov))
