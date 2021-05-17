import requests
from knock28 import remove_markup

dicts = remove_markup()
flag = dicts['国旗画像']
S = requests.Session()
URL = 'https://en.wikipedia.org/w/api.php'
PARAMS = {
    "action": "query",
    "format": "json",
    "prop": "imageinfo",
    "titles": 'File:' + flag,
    'iiprop' : 'url',
}

R = S.get(url=URL, params=PARAMS)
DATA = R.json()

PAGES = DATA["query"]["pages"]
print(PAGES[list(PAGES)[0]]['imageinfo'][0]['url'])