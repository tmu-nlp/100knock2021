import pandas as pd
import re
pattern = re.compile('File|ファイル:(.+?)\|')
wiki = pd.read_json('./data/jawiki-country.json.gz', lines = True, encoding='utf=8')
uk = wiki[wiki['title']=='イギリス'].text.values
ls = uk[0].split('\n')
for line in ls:
    r = re.findall(pattern, line)
    if r:
        print (r[0])