#Extract references to media files linked from the article.
import re

pattern = re.compile(r'.*?(File):(.+?)\|')

with open('/users/kcnco/github/100knock2021/pan/chapter03/x20.txt','r') as f:
    for line in f:
        result = re.search(pattern, line)
        if result:
            print(result.group(2))
