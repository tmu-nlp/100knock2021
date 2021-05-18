import re

with open('uk.json') as file:
    for line in file:
        line = line.replace('\n', '')
        line = re.findall(r'\[\[ファイル:(.*?)(?:\|.*)\]\]', line)
        if len(line) != 0:
            print(''.join(line))