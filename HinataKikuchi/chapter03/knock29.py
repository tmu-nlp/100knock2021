from knock28 import dict_info
import requests

url = 'https://www.mediawiki.org/w/api.php?' \
+ 'action=query' \
+ '&titles=File:' + dict_info['国旗画像'].replace(' ','_') \
+ '&format=json' \
+ '&prop=imageinfo' \
+ '&iiprop=url'

res = requests.get(url).json()

print(res['query']['pages']['-1']['imageinfo'][0]['url'])