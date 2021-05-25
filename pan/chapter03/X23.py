#Extract section names in the article with their levels. For example, the level of the section is 1 for the MediaWiki markup "== Section name ==".
import re

pattern = re.compile(r'(=+)(.+?)=+')

with open('/users/kcnco/github/100knock2021/pan/chapter03/x20.txt','r') as f:
    for line in f:
        result = re.match(pattern, line)
        if result:
            cnt = len(result.group(1))
            print( 'Section name:' + result.group(2) + 'level:' + str(cnt - 1))
