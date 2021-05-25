import json,re

with open('./jawiki-country.json','r',encoding='utf-8') as f:
    for line in f:
        line=json.loads(line)
        if line['title']=='イギリス':
            text=line['text']
            break

pattern='^(\={2,})\s*(.+?)\s*(\={2,}).*$'
result='\n'.join(i[1]+'='+str(len(i[0])) for i in re.findall(pattern,text,re.MULTILINE))
print(result)
print(re.findall(pattern,text,re.MULTILINE))

'''
{m,n} 直前の正規表現を m 回から n 回、できるだけ多く繰り返したものにマッチさせる結果の正規表現
m を省略すると下限は 0 に指定され、n を省略すると上限は無限に指定されます

\s Unicode (str)パターンでは Unicode 空白文字

+ 直前の正規表現を 1 回以上繰り返したもの
'''


