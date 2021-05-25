import pandas as pd
import re
import requests
pattern = re.compile('\|(.+?)\s=\s*(.+)')

wiki = pd.read_json('./data/jawiki-country.json.gz', lines = True, encoding='utf=8') 
uk = wiki[wiki['title']=='イギリス'].text.values

ls = uk[0].split('\n')
d = {}
for line in ls:
    r = re.search(pattern, line)
    if r:
        d[r[1]]=r[2]
        
S = requests.Session()
URL = "https://commons.wikimedia.org/w/api.php"
PARAMS = {
    "action": "query",
    "format": "json",
    "titles": "File:" + d['国旗画像'],
    "prop": "imageinfo",
    "iiprop":"url"
}
R = S.get(url=URL, params=PARAMS)
DATA = R.json()
PAGES = DATA['query']['pages']
for k, v in PAGES.items():
    print (v['imageinfo'][0]['url'])