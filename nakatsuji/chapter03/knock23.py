import re, json
from knock20 import text
for line in text.split('\n'):
    if re.match(r'^(={2,})\s*(.+?)\s*\1$', line):
        line = re.match(r'^(={2,})\s*(.+?)\s*\1$', line)
        k, n = line.group(2), len(line.group(1))-1
        print(f'{k} : {n}')