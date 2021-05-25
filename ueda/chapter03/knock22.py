import gzip
import json
import re
from knock20 import load_json

regex= re.compile(r'^\[\[Category:(.*?)(?:\|.*)*\]\]$', re.MULTILINE)
for cat in regex.findall(load_json()):
    print(cat)
