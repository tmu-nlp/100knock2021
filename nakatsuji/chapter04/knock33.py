from knock30 import sentences
#NP 名詞句
NPs = set()
for sen in sentences:
    for i in range(len(sen)):
        if sen[i]['surface'] == 'の' and i != len(sen)-1:

            prev = sen[i-1]
            next = sen[i+1]
            if prev['pos'] == '名詞' and next['pos'] == '名詞':
                np = prev['surface'] + 'の' + next['surface']
                NPs.add(np)
            

if __name__ == '__main__':
    
    print(NPs)
    print(len(NPs))