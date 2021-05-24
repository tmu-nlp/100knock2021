import re

import pandas as pd

pattern = re.compile('^=+.*=+$') # 1回以上の=で始まり、1回以上の=で終わる文字列
wiki = pd.read_json('./data/jawiki-country.json.gz', lines = True, encoding='utf=8')
uk = wiki[wiki['title']=='イギリス'].text.values
ls = uk[0].split('\n')
for line in ls:
    if re.search(pattern, line):
        level = line.count('=') // 2 - 1
        print(line.replace('=',''), level )