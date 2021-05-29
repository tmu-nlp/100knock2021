#記事中に含まれるセクション名とそのレベル（例えば”== セクション名 ==”なら1）を表示せよ．

#記事中でカテゴリ名を宣言している行を抽出せよ．

import json
import gzip
import re

f = gzip.open('jawiki-country.json.gz','rt')

for line in f:
    line = json.loads(line)
    if line['title'] == 'イギリス':
        text = line['text']
        break

pattern = r'^(\={2,})\s*(.+?)\s*(\={2,})$'

ans = re.findall(pattern, text, re.MULTILINE)

for a in ans:
    if a[0] != a[2]:
        print("error: Some thing wrong.")
        break
    
    name = a[1]
    level = len(a[0]) - 1
    print("{} : {}".format(name, level))

f.close()