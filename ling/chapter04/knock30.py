import sys
sentences=[]
morphs=[]

with open("./neko.txt.mecab",'r') as f:
    for line in f:
        
        if line != 'EOS\n':
            word_mor=line.split('\t')
            #print(word_mor[1])
            if word_mor[0]!='' or len(word_mor)==2:
                mor=word_mor[1].split(',')
                morph={'surface':word_mor[0],'base':mor[6],'pos':mor[0],'pos1':mor[1]}
                morphs.append(morph)
        else:
            sentences.append(morphs)
            morphs=[]

'''
for morph in sentences:
    print(morph)
'''

'''
elements in sentences look like:
{'surface': '平気', 'base': '平気', 'pos': '名詞', 'pos1': '形容動詞語幹'}
'''