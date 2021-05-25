#記事中に含まれる「基礎情報」テンプレートのフィールド名と値を抽出し，辞書オブジェクトとして格納せよ．

import json
import gzip
import re

f = gzip.open('jawiki-country.json.gz','rt')

for line in f: #イギリスのページを抽出
    line = json.loads(line)
    if line['title'] == 'イギリス':
        text = line['text']
        break

pattern = r'^\{\{基礎情報.*?$(.*?)^\}\}' #基礎情報の項目は{{基礎情報 <内容＞ \n}}で表されているので^\}}で締めくくれる

info = re.findall(pattern, text, re.MULTILINE + re.DOTALL)

for a in info: #基礎情報の抽出
    pattern2 = r'^\|(.+?)\s*\=\s*(.+?)(?:(?=\n\|)|(?=\n$))'
    ans_dict = dict(re.findall(pattern2, a, re.MULTILINE + re.DOTALL))

for key, value in ans_dict.items():
    print("{} : {}".format(key, value))

f.close()