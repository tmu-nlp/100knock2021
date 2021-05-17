#In addition to the process of the problem 25, remove emphasis MediaWiki markups from the values.
import re
import json
import gzip

pattern = re.compile(r'^\|(.+?)\s=\s(.+?)(\n)', re.MULTILINE + re.DOTALL)
pattern_1 = re.compile(r'\'{2,5}', re.MULTILINE + re.DOTALL)

dict  = {}

with gzip.open('/users/kcnco/github/100knock2021/pan/chapter03/enwiki-country.json.gz', 'r') as country_f:
    for line in country_f:
        line = json.loads(line)
        if line['title'] == 'United Kingdom':
            for ans in pattern.finditer(line['text']):
                ans2 = pattern_1.sub("", ans.group(2))
                dict[ans.group(1)] = ans2

for key, value in dict.items():
    print(key + ":" + value)
