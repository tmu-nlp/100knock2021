#Extract field names and their values in the Infobox “country”, and store them in a dictionary object.
import re
import json
import gzip

pattern = re.compile(r'^\|(.+?)\s=\s(.+?)\n', re.MULTILINE + re.DOTALL)

dict = {}

with gzip.open('/users/kcnco/github/100knock2021/pan/chapter03/enwiki-country.json.gz', 'r') as country_f:
    for line in country_f:
        line = json.loads(line)
        if line['title'] == 'United Kingdom':
            for ans in re.findall(pattern, line['text']):
                dict[ans[0]] = ans[1]

for key, value in dict.items():
    print(key + ":" + value)
