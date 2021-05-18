import re
from collections import defaultdict

with open('uk.json') as file:
    template = list()
    i = 0
    for line in file:
        line = line.replace('\n', '')
        if 3 <= i <= 68:
            template.append(line)
        if i == 69:
            break
        i += 1

dic = defaultdict(lambda: 0)
for line in template:
    #^\|・・・先頭が|
    if re.match(r'^\|(.*?)', line):
        #|と:の間を抽出
        field = re.findall(r'\|(.*?)(?:\=.*)', line)
        #:以降を抽出
        value = re.findall(r'(?:\.*=)(.*)', line)
        #knock25に追加
        value = re.sub(r"'{2,5}", '', ''.join(value))
        #knock26に追加
        value = re.sub(r'\[\[|\]\]', '', ''.join(value))

        dic[''.join(field)] = ''.join(value)

for k in dic:
    print(f"{k}:{dic[k]}")