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
    p='\'{2,5}'
    t=re.sub(p,'',t)
    return t

result={k:rm(v) for k,v in dic.items()}
for key,value in result.items():
    print(key+':'+value+'\n')

'''
re.sub(pattern, repl, string, count=0, flags=0)
string 中に出現する最も左の重複しない pattern を置換 repl で置換することで得られる文字列を返します

'''