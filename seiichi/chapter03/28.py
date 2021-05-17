import json, re
from common import load_json
j = load_json()
m1 = re.search(r'{{基礎情報 国.*?', j)
m2 = re.search(r'(.*)\n}}\n', j[m1.end():])
text = j[m1.end():m2.end()+1].split('\n')
d = {}
for line in text:
    m = re.match(r'\|(.+)( = )(.*)', line)
    if m is None: continue
    m3 = re.match(r"(.*)('{3,5})(.*?)(\2)(.*)", m.group(3).strip())
    if m3 is None: tmp = line
    else: tmp = m3.group(1) + m3.group(3) + m3.group(5)
    tmp = re.sub(r"\[\[(?:[^:\]]+?\|)?([^:]+?)\]\]", r"\1", tmp)
    tmp = re.sub(r"<.+?>", r"", tmp)
    d[m.group(1).strip()] = tmp
#print(d)
for k, v in d.items():
    print(k, v)

