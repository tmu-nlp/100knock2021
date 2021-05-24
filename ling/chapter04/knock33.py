from knock30 import sentences

ans=set() 
for sentence in sentences:
    for i in range(1,len(sentence)-1):
        if sentence[i-1]['pos']=='名詞' and sentence[i]['surface']=='の' and sentence[i+1]['pos']=='名詞':
            ans.add(sentence[i-1]['surface']+sentence[i]['surface']+sentence[i+1]['surface'])        
            

[print(v) for v in list(ans)]
print(len(ans))