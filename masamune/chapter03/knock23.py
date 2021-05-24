#レベル：「=」の長さ - 1
import re

with open('uk.json') as file:
    for line in file:
        line = line.replace('\n', '')
        #^ 行の先頭
        if re.match(r'^(={2,}).*(={2,})', line):
            section_name = line.replace('=', '')
            level = line.count('=')//2 - 1
            print(f'{section_name}：{level}')