from knock30 import load_mecab

verb = set()
for line in load_mecab():
    for morpheme in line:
        if(morpheme['pos'] == '動詞'):
            verb.add(morpheme['base'])
print(verb)