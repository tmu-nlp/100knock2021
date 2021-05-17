import pandas as pd
import re


df = pd.read_json('./data/jawiki-country.json.gz', lines = True, encoding='utf=8')
uk_text = df.query('title=="イギリス"')['text'].values[0]
uk_texts = uk_text.split('\n')

pattern = re.compile('\|(.+?)\s=\s*(.+)')
ans = {}
for line in uk_texts:
    r = re.search(pattern, line)
    if r:
        ans[r[1]] = r[2]
print(ans)