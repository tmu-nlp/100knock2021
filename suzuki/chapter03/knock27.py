#26の処理に加えて，テンプレートの値からMediaWikiの内部リンクマークアップを除去し，テキストに変換せよ（参考: マークアップ早見表）．

import json
import gzip
import re

def remove_emp(text):
    pattern = r'\'{2,5}'
    text = re.sub(pattern, '', text)
    return text

def remove_link(text):
    pattern = r'\[\[(?:[^|]*?\|)??([^|]*?)\]\]'
    text = re.sub(pattern, r'\1', text)
    return text

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
    value = remove_emp(value)
    value = remove_link(value)
    print("{} : {}".format(key, value))

f.close()