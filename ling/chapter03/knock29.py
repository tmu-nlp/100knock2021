import json,re,requests

with open('./jawiki-country.json','r',encoding='utf-8') as f:
    for line in f:
        line=json.loads(line)
        if line['title']=='イギリス':
            text=line['text']
            break

pattern='^\{\{基礎情報.*?$(.*?)^\}\}'
temp=re.findall(pattern,text,re.MULTILINE|re.DOTALL)
pattern = '^\|(.+?)\s*=\s*(.+?)(?:(?=\n\|)|(?=\n$))'

dic=dict(re.findall(pattern,temp[0],re.MULTILINE+re.DOTALL))

urlfile=dic['国旗画像'].replace(' ','_')
url = 'https://commons.wikimedia.org/w/api.php?action=query&titles=File:' + urlfile + '&prop=imageinfo&iiprop=url&format=json'

data=requests.get(url).text
#print(type(link))
link=re.search('"url":"(.+?)"',data).group(1)
print(link)

