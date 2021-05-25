import re

with open('uk.json') as file:
    for line in file:
        line = line.replace('\n', '')
        #.改行以外 *直前の正規表現を繰り返す \特殊シーケンスの合図
        if re.match(r'\[\[Category:.*\]\]', line):
            print(line)