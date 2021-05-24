#27の処理に加えて，テンプレートの値からMediaWikiマークアップを可能な限り除去し，国の基本情報を整形せよ．

import json
import gzip
import re

def remove_emp(text): #26 強調マークアップの除去
    pattern = r'\'{2,5}'
    text = re.sub(pattern, '', text)
    return text

def remove_link(text): #27 内部リンクの除去
    pattern = r'\[\[(?:[^|]*?\|)??([^|]*?)\]\]'
    text = re.sub(pattern, r'\1', text)
    return text

def remove_mediawiki(text): #28 残りのマークアップの除去
    pattern = r'<.+?>'
    text = re.sub(pattern, '', text)

    pattern = r'\{\{Cite web\s*\|.*?\}\}'
    text = re.sub(pattern, '', text)

    pattern = r'\[http:.+?\]'
    text = re.sub(pattern, '', text)

    pattern = r'\{.\}'
    text = re.sub(pattern, ' ', text)

    return text

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
    value = remove_emp(value)
    value = remove_link(value)
    value = remove_mediawiki(value)
    print("{} : {}".format(key, value))

f.close()