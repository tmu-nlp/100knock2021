from knock30 import parse_mecab
from pprint import pprint


with open('neko.txt.mecab') as f:
    parsed = parse_mecab(f.readlines())
res, t_s, t_c = [], '', 0
for morph in parsed:
    if morph['pos'] == '名詞':
        t_s += morph['surface']
        t_c += 1
    else:
        if t_c > 1:
            res.append(t_s)
        t_s = ''
        t_c = 0
pprint(res)
