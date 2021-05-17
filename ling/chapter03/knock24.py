import json,re

with open('./jawiki-country.json','r',encoding='utf-8') as f:
    for line in f:
        line=json.loads(line)
        if line['title']=='イギリス':
            text=line['text']
            break

pattern='\[\[ファイル:(.+?)\|'
result='\n'.join(re.findall(pattern,text))
print(result)
#print(re.findall(pattern,text,re.MULTILINE))

#[[ファイル:Soldiers Trooping the Colour, 16th June 2007.jpg|thumb|軍旗分列行進式における[[近衛兵 (イギリス)|近衛兵]]]]