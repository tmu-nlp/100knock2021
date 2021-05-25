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

print(dic['国旗画像']+'\n')

urlfile=dic['国旗画像'].replace(' ','_')
url = 'https://commons.wikimedia.org/w/api.php?action=query&titles=File:' + urlfile + '&prop=imageinfo&iiprop=url&format=json'
print(url+'\n')

data=requests.get(url).text
print(type(requests.get(url)))

print(type(re.search('"url":"(.+?)"',data)))
print(re.search('"url":"(.+?)"',data))

link=re.search('"url":"(.+?)"',data).group(1)
print(link)

'''
requests.get()
    引数:url
    返り値：Response object
    テキスト: text属性

*?, +?, ??
非貪欲あるいは 最小のマッチが行われ、できるだけ少ない文字にマッチ

'''
