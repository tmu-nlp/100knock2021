#25の処理時に，テンプレートの値からMediaWikiの強調マークアップ（弱い強調，強調，強い強調のすべて）を除去してテキストに変換せよ（参考: マークアップ早見表）．

import json
import gzip
import re

f = gzip.open('jawiki-country.json.gz','rt')

for line in f:
    line = json.loads(line)
    if line['title'] == 'イギリス':
        text = line['text']
        break

pattern = r'^\{\{基礎情報.*?$(.*?)^\}\}' #基礎情報の項目は{{基礎情報 <内容＞ \n}}で表されているので^\}}で締めくくれる

info = re.findall(pattern, text, re.MULTILINE + re.DOTALL)

for a in info:
    pattern2 = r'^\|(.+?)\s*\=\s*(.+?)(?:(?=\n\|)|(?=\n$))'
    ans_dict = dict(re.findall(pattern2, a, re.MULTILINE + re.DOTALL))

for key, value in ans_dict.items():
    pattern3 = r'\'{2,5}'
    value = re.sub(pattern3, '', value)
    print("{} : {}".format(key, value))

f.close()