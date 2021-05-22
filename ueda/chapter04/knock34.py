from knock30 import load_mecab

n_concat = set()
for line in load_mecab():
    tmp = []
    for morpheme in line:
        if morpheme['pos'] == '名詞':
            tmp.append(morpheme['surface'])
        else:
            if len(tmp) >= 2:
                n_concat.add("".join(tmp))
            tmp=[]
print(n_concat)