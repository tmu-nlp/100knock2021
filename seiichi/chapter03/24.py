import json, re
from common import load_json
j = load_json()
for line in j.split('\n'):
    if re.match(r'\[\[:*(ファイル|File):.*\]\]$', line):
        print(re.match(r'\[\[:*(ファイル|File):(.+?)\|.+\]\]$', line).group(2))
    elif re.match(r'^(ファイル|File):.*\]\]$', line):
        print(re.match(r'^(ファイル|File):(.+?)\|.+$', line).group(2))

