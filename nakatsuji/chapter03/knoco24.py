import re, json
from knock20 import text
for line in text.split('\n'):
    if re.match(r'\[\[ファイル:(.+?)', line):
        file = re.match(r'\[\[ファイル:(.+?)\|', line)
        print(file.group(1))