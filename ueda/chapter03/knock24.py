import gzip
import json
import re
from knock20 import load_json

regex= re.compile(r'^\[\[ファイル:(.*?)\|.*\]\]$', re.MULTILINE)
for media in regex.findall(load_json()):
    print(media)
