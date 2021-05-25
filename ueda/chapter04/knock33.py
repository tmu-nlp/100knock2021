from knock30 import load_mecab

n_p = set()
for line in load_mecab():
    for i in range(0, len(line)-2):
        if line[i]['pos'] == '名詞' and line[i+1]['base'] == 'の' and line[i+2]['pos'] == '名詞':
            n_p.add(line[i]['surface']+line[i+1]['surface']+line[i+2]['surface'])
print(n_p)