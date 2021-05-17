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
for key,value in dic.items():
    print(key+':'+value+'\n')
'''
re.DOTALL 特殊文字 '.' を、改行を含む任意の文字と、とにかくマッチさせます；
このフラグがなければ、 '.' は、改行 以外の 任意の文字とマッチします。


'''