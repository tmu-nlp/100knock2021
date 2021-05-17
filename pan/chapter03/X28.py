#In addition to the process of the problem 27, remove MediaWiki markups from the values as much as you can, and obtain the basic information of the country in plain text format.
import re
import json
import gzip

pattern_0 = re.compile(r'^\|(.+?)\s=\s(.+?)(\n)', re.MULTILINE + re.DOTALL)
pattern_1 = re.compile(r'\'{2,5}', re.MULTILINE + re.DOTALL)
pattern_2 = re.compile(r'\[{2}(?:[^|]*?\|)*?([^|]*?)\]{2}', re.MULTILINE + re.DOTALL)
pattern_3 = re.compile(r'\{\{lang(?:[^|]*?\|)([^|]*?)\}\}', re.MULTILINE + re.DOTALL)
pattern_4 = re.compile(r'<\/?[br|ref](.*?)>', re.MULTILINE + re.DOTALL)

dict  = {}

with gzip.open('/users/kcnco/github/100knock2021/pan/chapter03/enwiki-country.json.gz', 'r') as country_f:
    for line in country_f:
        line = json.loads(line)
        if line['title'] == 'United Kingdom':
            for ans in pattern_0.finditer(line['text']):
                ans2 = pattern_1.sub("", ans.group(2))
                ans2 = pattern_2.sub(r'\1', ans2)
                ans2 = pattern_3.sub(r'\1', ans2)
                ans2 = pattern_4.sub('', ans2)
                dict[ans.group(1)] = ans2

for key, value in dict.items():
    print(key + ":" + value)
