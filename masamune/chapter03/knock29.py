import re
from collections import defaultdict
import requests

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
        dic[''.join(field)] = ''.join(value)

def get_url(dic):
    url_file = str(dic['国旗画像 ']).replace(' ', '_')
    url = 'https://commons.wikimedia.org/w/api.php?action=query&titles=File:' + url_file + '&prop=imageinfo&iiprop=url&format=json'
    data = requests.get(url)
    #dataのレスポンスからurlを取り出す。search()でマッチ、group()でマッチした文字列を取得
    return re.search(r'"url":"(.+?)"', data.text).group(1)

print(get_url(dic))