import json,re

with open('./jawiki-country.json','r',encoding='utf-8') as f:
    for line in f:
        line=json.loads(line)
        if line['title']=='イギリス':
            text=line['text']
            break

pattern='^\{\{基礎情報.*?$(.*?)^\}\}'
temp=re.findall(pattern,text,re.MULTILINE|re.DOTALL)
#print(temp)

pattern = '^\|(.+?)\s*=\s*(.+?)(?:(?=\n\|)|(?=\n$))'
#print(re.findall(pattern,temp[0],re.MULTILINE+re.DOTALL))
dic=dict(re.findall(pattern,temp[0],re.MULTILINE+re.DOTALL))
def rm(t):
    #remove Character formatting
    p='\'{2,5}(.*?)(\1)'
    t=re.sub(p,'\2',t)
    #remove internal link and file
    p='\[\[(?:[^|]*?\|)*?([^|]*?)\]\]'
    t=re.sub(p,'\1',t)
    #remove {{}}
    #p='\{\{(?:[^|]*?\|)*?([^|]*?)\}\}'
    #t=re.sub(p,'\1',t)
    #remove tag
    p='<\/?[br|ref][^>]*?>'
    t=re.sub(p,'',t)
    #remove [http://xxxx] [http://xxx xxx]
    p='\[http:\/\/(?:[^\s]*?\s)?([^]]*?)\]'
    t=re.sub(p,'\1',t)
    #remove {{lang|tag|string}}
    p='\{\{lang(?:[^|]*?\|)*?([^|]*?)\}\}'
    t=re.sub(p,'\1',t)
    return t
    

result={k:rm(v) for k,v in dic.items()}
for key,value in result.items():
    print(key+':'+value+'\n')

'''


'''