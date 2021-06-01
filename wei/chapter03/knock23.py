'''23. セクション構造
記事中に含まれるセクション名とそのレベル（例えば”== セクション名 ==”なら1）を表示せよ．'''


import re
from knock20 import read_gzip

if __name__ == '__main__':
    filepath = './data/jawiki-country.json.gz'
    text = read_gzip(filepath, 'イギリス')

    pattern = r'=+.*?=+?\n'
    sections = re.findall(pattern, text)     # list
    # print(''.join(sections).strip(''))
    for line in sections:
        level = int(line.count('=')/2) - 1

        print(line.replace('\n','').replace('=','').strip(),' ',level)


'''国名 1
歴史 1
地理 1
主要都市 2
気候 2
政治 1
元首 2
法 2'''