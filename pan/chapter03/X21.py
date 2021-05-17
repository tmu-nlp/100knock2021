#Extract lines that define the categories of the article.
import re
import gzip
import json

pattern = re.compile(r'.*(\[\[Category:.*\]\]).*')

with open('/users/kcnco/github/100knock2021/pan/chapter03/x20.txt','r') as f:
    for line in f:
        result = re.match(pattern, line)
        if result:
            print(result.group())
