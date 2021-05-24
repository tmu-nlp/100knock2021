from knock30 import sentences

ans=set()

for sentence in sentences:
    noun=''
    length=0
    for mor in sentence:
        if mor['pos']=='名詞':
            noun=''.join([noun,mor['surface']])
            length+=1
        elif length>1:
            ans.add(noun)
            noun=''
            length=0
        else:
            noun=''
            length=0
        
    if length>1:
        ans.add(noun)

for v in ans:
    print(v+'\n')
print(len(ans))
