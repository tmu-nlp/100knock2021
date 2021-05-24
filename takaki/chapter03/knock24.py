from knock20 import load_england
import re

PATTERN = re.compile(r'\[\[ファイル:(.*?)(?:\|.*)?\]\]')

for match in re.findall(PATTERN, load_england()):
    print(match)
