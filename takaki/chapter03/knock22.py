from knock20 import load_england
import re

PATTERN = re.compile(r'^\[\[Category:(.*?)(?:\|.*)?\]\]$')

for line in load_england().split('\n'):
    match = re.match(PATTERN, line)
    if match != None:
        print(match.group(1))
