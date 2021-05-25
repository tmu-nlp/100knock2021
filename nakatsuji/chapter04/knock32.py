from knock30 import sentences
from knock31 import verb_sur
verb_base = set()
for sen in sentences:
    for word in sen:
        if word['pos'] == '動詞':
            verb_base.add(word['base'])

if __name__ == '__main__':
    #比較
    print(len(verb_base), len(verb_sur))
    print(verb_sur)