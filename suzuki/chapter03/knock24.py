#記事から参照されているメディアファイルをすべて抜き出せ．

import json
import gzip
import re

f = gzip.open('jawiki-country.json.gz','rt')

for line in f:
    line = json.loads(line)
    if line['title'] == 'イギリス':
        text = line['text']
        break

pattern = r'\[\[ファイル:(.+?)\|'

ans = re.findall(pattern, text, re.MULTILINE)

for a in ans:
    print(a)

f.close()