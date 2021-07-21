""'''
28. MediaWikiマークアップの除去
27の処理に加えて，テンプレートの値からMediaWikiマークアップを可能な限り除去し，
国の基本情報を整形せよ．'''

import re
from knock25 import basic_dict
from knock27 import remove_link

def remove_markups(x):
    x = re.sub(r'{{.*\|.*\|([^}]*)}}', r'\1', x)
    x = re.sub(r'<([^>]*)( .*|)>.*</\1>', '', x)
    x = re.sub(r'\{\{0\}\}', '', x)
    return x


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
    for k, v in dict4.items():
        print(k, v)


