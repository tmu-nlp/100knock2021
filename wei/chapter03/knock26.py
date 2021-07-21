""'''26. 強調マークアップの除去
25の処理時に，テンプレートの値からMediaWikiの強調マークアップ
（弱い強調，強調，強い強調のすべて）を除去してテキストに変換せよ

[ref]
- re.sub(pattern, repl, string, count=0, flags=0)
  - 返回通过使用repl 替换在 string 最左边非重叠出现的 pattern而获得的字符串。
    若pattern没找到，则不加改变地返回string。 
  '''

import re
from knock25 import basic_dict


if __name__ == '__main__':

    file = './data/UK_info.json'
    pattern = r'\|(.*?)\s=\s*(.+)'

    '''|国旗画像 = Flag of the United Kingdom.svg
    |国章画像 = [[ファイル:Royal Coat of Arms of the United Kingdom.svg|85px|イギリスの国章]]
    |国章リンク =（[[イギリスの国章|国章]]）
    ...'''

    basic_info = basic_dict(file, pattern)
    # 強調マークアップを除去して、辞書に保持
    dict2 = {
        key : re.sub(r"''+", '',value)
        for key, value in basic_info.items()
    }
    for k, v in dict2.items():
        print(k, v)

