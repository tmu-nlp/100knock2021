import json, re
from common import load_json
j = load_json()
for line in j.split('\n'):
    if re.match(r'^\[\[Category:.+\]\]$', line):
        c = re.match(r'^\[\[Category:(.*)\]\]$', line)
        print(c.group(1).split('|')[0])
