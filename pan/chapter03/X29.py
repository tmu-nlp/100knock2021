######

#Obtain the URL of the country flag by using the analysis result of Infobox.
####
import gzip
import json
import re
import urllib.parse, urllib.request

pattern = re.compile(r'^\|(.+?)\s=\s(.+?)(\n)', re.MULTILINE + re.DOTALL)

dict = {}

with gzip.open('/users/kcnco/github/100knock2021/pan/chapter03/enwiki-country.json.gz', 'r') as country_f:
    for line in country_f:
        line = json.loads(line)
        if line['title'] == 'United Kingdom':
            for ans in pattern.finditer(line['text']):
                dict[ans.group(1)] = ans.group(2)
name_flag = dict['????????']

url = 'https://en.wikipedia.org/w/api.php' \
    + 'action=query' \
    + '&titles=File:' + urllib.parse.quote(name_flag) \
    + '&format=json' \
    + '&prop=imageinfo' \
    + '&iiprop=url'

request = urllib.request.Request(url)
connection = urllib.request.urlopen(request)
data = json.loads(connection.read().decode())
url = data['query']['pages'].popitem()[1]['imageinfo'][0]['url']
print(url)
