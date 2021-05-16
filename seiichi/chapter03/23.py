import json, re
from common import load_json
j = load_json()
for line in j.split('\n'):
    if re.match(r'^=+.+=+$', line):
        m = re.match(r'^(=+)(.+?)=+$', line)
        print(' name : {0}\tlevel : {1} '.format(m.group(2), len(m.group(1)) - 1))
