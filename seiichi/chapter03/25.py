import json, re
from common import load_json
j = load_json()
m1 = re.search(r'{{基礎情報 国.*?', j)
m2 = re.search(r'(.*)\n}}\n', j[m1.end():])
text = j[m1.end():m2.end()+1].split('\n')
d = {}
for line in text:
    m = re.match(r'\|(.+)=(.*)', line)
    if m is None: continue
    d[m.group(1).strip()] = m.group(2).strip()
print(d)

