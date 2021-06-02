from itertools import combinations
import re
from knock41 import load_chunk

f = open(r'/Users/Naoya/Downloads/nounphrase.txt', 'w', encoding="utf-8")
for line in load_chunk():
    noun_list = []
    for l in line:
        if '名詞' in [tmp.pos for tmp in line[l].morphs] and int(line[l].dst) != -1:
            noun_list.append(l)
    for i,j in combinations(noun_list, 2):
        ipaths = []
        jpaths = []
        while i != j:
            if int(i) < int(j):
                ipaths.append(i)
                i = line[i].dst
            else:
                jpaths.append(j)
                j = line[j].dst
        Xline = []
        Yline = []
        #名詞が2重になる場合があるから除去したい
        for word in line[ipaths[0]].morphs:
            if word.pos == '名詞':
                Xline.append('#')
            elif word.pos!= '記号':
                Xline.append(word.surface)
        if len(jpaths) == 0:
            for word in line[i].morphs:
                if word.pos == '名詞':
                    Yline.append('#')
                elif word.pos!= '記号':
                    Yline.append(word.surface)
            Zline = [''.join(word.surface for word in line[id].morphs if word.pos != '記号') for id in ipaths[1:]]
            Xline = re.sub('#+', 'X', ''.join(Xline))
            Yline = re.sub('#+', 'Y', ''.join(Yline))
            f.write(' -> '.join([Xline]+ Zline + [Yline])+'\n')
        else:
            for word in line[jpaths[0]].morphs:
                if word.pos == '名詞':
                    Yline.append('#')
                elif word.pos!= '記号':
                    Yline.append(word.surface)
            Xline = re.sub('#+', 'X', ''.join(Xline))
            Yline = re.sub('#+', 'Y', ''.join(Yline))
            Xline = [Xline] + [''.join(word.surface for word in line[id].morphs if word.pos != '記号') for id in ipaths[1:]]
            Yline = [Yline] + [''.join(word.surface for word in line[id].morphs if word.pos != '記号') for id in jpaths[1:]]
            Zline = ''.join([word.surface for word in line[i].morphs])
            f.write(' | '.join([' -> '.join(Xline), ' -> '.join(Yline), Zline])+'\n')



