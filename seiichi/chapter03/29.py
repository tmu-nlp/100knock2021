import json, requests, re
from common import load_json
j = load_json()
res = dict(re.findall(r'\|(.+?)\s=\s*(.+)', j, re.MULTILINE))
s = requests.Session()
url = "https://en.wikipedia.org/w/api.php"
params = {
    "action": "query",
    "format": "json",
    "prop": "imageinfo",
    "titles": f"File:{res['国旗画像']}",
    "iiprop": "url"
}
ret = s.get(url=url, params=params)
data = ret.json()
data_dict = data["query"]["pages"]["23473560"]["imageinfo"][0]
print(data_dict['url'])
