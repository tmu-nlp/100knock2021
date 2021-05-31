from knock41 import sentences
from knock42 import dependencies

pattern_V_case = []
with open('output45.txt', 'w') as out:
    for sentence in sentences:
        #係り先が動詞かの否かのフラグ
        flag = 0


        for chunk in sentence:
            verb = ''
            for morph in chunk.morphs:
                if morph.pos == '動詞':
                    verb = morph.base
                    break
            if verb == '':
                continue
            cases = []
            for src in chunk.srcs:
                for morph in sentence[src].morphs:
                    if morph.pos == '助詞':
                        cases.append(morph.base)             
            cases.sort()

            writing = f'{verb}\t' + ' '.join(cases)
            out.write(writing + '\n')
        


'''
cat output45.txt | sort | uniq -c | sort -n -r | head
49 する       を
  18 する       が
  15 する       に
  14 する       と
  12 する       は を
  10 する       に を
   9 する       で を
   9 よる       に
   8 行う       を
   8 する
cat output45.txt | grep '^行う\s'| sort | uniq -c | sort -n -r | head
   8 行う       を
   1 行う       まで を
   1 行う       から
   1 行う       に まで を
   1 行う       は を をめぐって
   1 行う       に に により を
   1 行う       て に は は は
   1 行う       が て で に は
   1 行う       が で に は
   1 行う       に を を
cat output45.txt | grep '^なる\s'| sort | uniq -c | sort -n -r | head
   3 なる       に は
   3 なる       が と
   2 なる       に
   2 なる       と
   1 なる       から が て で と は
   1 なる       から で と
   1 なる       て として に は
   1 なる       が と にとって は
   1 なる       で と など は
   1 なる       が で と に は は
cat output45.txt | grep '^与える\s'| sort | uniq -c | sort -n -r | head
   1 与える     が など に
   1 与える     に は を
   1 与える     が に
'''
