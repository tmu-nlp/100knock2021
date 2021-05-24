import gzip
import json
import re
from knock20 import load_json

regex= re.compile(r'^(={2,})\s*(.+?)\s*\1$', re.MULTILINE)
for sec in regex.findall(load_json()):
    print(sec[1]+" "+str(len(sec[0])-1))
