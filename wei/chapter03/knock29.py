'''29. 国旗画像のURLを取得する
テンプレートの内容を利用し，国旗画像のURLを取得せよ．'''""

import re
from knock25 import basic_dict
from knock27 import remove_link
from knock28 import remove_markups
import requests


def get_url(text):
    url_file = text['国旗画像'].replace(' ', '_')
    url = 'https://commons.wikimedia.org/w/api.php?action=query&titles=File:' + url_file + '&prop=imageinfo&iiprop=url&format=json'
    data = requests.get(url)
    return re.search(r'"url":"(.+?)"', data.text).group(1)


if __name__ == '__main__':

    file = './data/UK_info.json'
    pattern = r'\|(.*?)\s=\s*(.+)'
    basic_info = basic_dict(file, pattern)
    dict2 = {
        key : re.sub(r"''+", '',value)
        for key, value in basic_info.items()
    }

    dict3 = {
        key : remove_link(value)
        for key, value in dict2.items()
    }

    dict4 = {
        key : remove_markups(value)
        for key, value in dict3.items()
    }

    res = get_url(dict4)

    print(res)
