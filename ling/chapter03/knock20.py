import json

with open('./jawiki-country.json','r') as f:
    for line in f:
        line=json.loads(line)
        if line['title']=='イギリス':
            text=line
    print(text)