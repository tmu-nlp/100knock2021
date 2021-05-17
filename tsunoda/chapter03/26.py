import pandas as pd
import re

pattern = re.compile('\|(.+?)\s=\s*(.+)')
p_emp = re.compile('\'{2,}(.+?)\'{2,}')
wiki = pd.read_json('./data/jawiki-country.json.gz', lines = True, encoding='utf=8') 
uk = wiki[wiki['title']=='イギリス'].text.values
ls = uk[0].split('\n')
d = {}
for line in ls:
    r = re.search(pattern, line)
    if r:
        d[r[1]]=r[2]
    r = re.sub(p_emp,'\\1', line)
    
print (d)