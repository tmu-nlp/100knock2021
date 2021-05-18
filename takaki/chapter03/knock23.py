from knock20 import load_england
import re

PATTERN = re.compile(r'^(\={2,})\s*(.+?)\s*(\={2,})$')

for line in load_england().split('\n'):
    match = re.match(PATTERN, line)
    if match != None and len(match.group(1)) == len(match.group(3)):
        print(f"Level:{len(match.group(1))-1} / Section: {match.group(2)}")
