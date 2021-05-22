from knock30 import sentences

verb_sur = set()
lis = []
for sen in sentences:
    for word in sen:
        if word['pos'] == '動詞':
            verb_sur.add(word['surface'])
            lis.append(word['surface'])
if __name__ == '__main__':
    #比較
    print(len(lis), len(verb_sur))
    print(verb_sur)