'''27. 内部リンクの除去
26の処理に加えて，テンプレートの値からMediaWikiの内部リンクマークアップを除去し，
テキストに変換せよ'''""

import re
from knock25 import basic_dict

# 内部リンクマークアップを除去
def remove_link(x):
    x = re.sub(r'\[\[[^\|\]]+\|[^{}\|\]]+\|([^\]]+)\]\]', r'\1', x)
    x = re.sub(r'\[\[[^\|\]]+\|([^\]]+)\]\]', r'\1', x)
    x = re.sub(r'\[\[([^\]]+)\]\]', r'\1', x)
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

    for k, v in dict3.items():
        print(k, v)