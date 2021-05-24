import pandas as pd
import re


df = pd.read_json('./data/jawiki-country.json.gz', lines = True, encoding='utf=8')
uk = df.query('title=="イギリス"')['text'].values[0]
ls = uk.split('\n')
pattern = r"\[\[Category:(.*)\]\]"

for line in ls:
	if re.match(pattern, line):
		line = re.sub(r"\[\[Category:", " ", line)
		line = re.sub(r"(\|.*)*(\|\*)*(\]\])", " ", line)
		print(line)

#r"\[\[Category:(?P<Category_name>[^|]+)\|*(?P<Sortkey>.*)\]\]"
#[[Category:カテゴリ名]] or [[Category:カテゴリ名|ソートキー]]
	   