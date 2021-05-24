import re
import json
from knock20 import text
for line in text.split('\n'):
    if re.match(r'^\[\[Category:.*\]\]$', line):
        print(line)
