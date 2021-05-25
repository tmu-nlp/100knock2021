from knock30 import sentences

ans=set() 
for sentence in sentences:
    for morph in sentence:
        if morph['pos']=='動詞':
            ans.add(morph['base'])

[print(v) for v in list(ans)]
print(len(ans))