import pandas as pd
import re

df = pd.read_json('./data/jawiki-country.json.gz', lines = True, encoding='utf=8')
uk = df[df.title=="イギリス"].text.values[0]
for category in re.findall(r"\[\[Category:.*?\]\]", uk):
    print(category)