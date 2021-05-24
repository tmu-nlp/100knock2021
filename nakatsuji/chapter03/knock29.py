import requests
from knock28 import dic_remove_mwmu


S = requests.Session()

URL = "https://en.wikipedia.org/w/api.php"

PARAMS = {
    "action": "query",
    "format": "json",
    "prop": "imageinfo",
    "titles": "File:" + dic_remove_mwmu['国旗画像'],
    'iiprop' : 'url',
}

R = S.get(url=URL, params=PARAMS)
DATA = R.json()

PAGES = DATA["query"]["pages"]
#for k, v in PAGES.items():
#    print(f'{k} >>> {v}')
print(PAGES[list(PAGES)[0]]['imageinfo'][0]['url'])