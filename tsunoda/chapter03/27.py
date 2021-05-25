import pandas as pd
import re

pattern = re.compile('\|(.+?)\s=\s*(.+)')
p_emp = re.compile('\'{2,}(.+?)\'{2,}')
p_link = re.compile('\[\[(.+?)\]\]')
wiki = pd.read_json('./data/jawiki-country.json.gz', lines = True, encoding='utf=8') 
uk = wiki[wiki['title']=='イギリス'].text.values
lines = uk[0]
lines = re.sub(p_emp,'\\1', lines)
lines = re.sub(p_link,'\\1', lines)
ls = uk[0].split('\n')
d = {}
for line in ls:
    r = re.search(pattern, line)
    if r:
        d[r[1]]=r[2]

    
print (d)