#記事のカテゴリ名を（行単位ではなく名前で）抽出せよ．

import json
import gzip
import re

f = gzip.open('jawiki-country.json.gz','rt')

for line in f:
    line = json.loads(line)
    if line['title'] == 'イギリス':
        text = line['text']
        break

pattern = r'^.*\[\[Category:(.*?)(?:\|.*)?\]\].*$'
# r : raw文字列記法にする
# ^ : 文字列の先頭，改行の直後
# () : グループの開始と終了を表す
# \[ : 文字列として[]を表現
# ? : 直前の正規表現を0回または1回繰り返したものにマッチ
# .* : 任意の0文字以上の文字列
# (?:文字列) : '?:'以降の文字列を敢えて取得しない
# $ : 文字列の末尾，改行の直前

# \[\[Category:.*\]\] の前後にも .* がついている理由はカテゴリのみについてではなく，行全体を抽出するため．

ans = re.findall(pattern, text, re.MULTILINE)

for a in ans:
    print(a)

f.close()