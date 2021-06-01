""'''
25. テンプレートの抽出
記事中に含まれる「基礎情報」テンプレートのフィールド名と値を抽出し，
辞書オブジェクトとして格納せよ．
{{基礎情報 国　
|key1 = value1
|key2 = value2 
...
}}
'''

import re


def basic_dict(filename, pattern):
    with open(filename, 'r', encoding='utf-8') as f:
        dict = {}
        flag_start = False
        for line in f:
            if re.search(r'{{基礎情報\s*国',line):
                flag_start = True
                continue
            if flag_start:
                if re.search(r'^}}$', line):
                    break
            templete = re.search(pattern, line)
            if templete:
                key = templete.group(1).strip()
                dict[key] = templete.group(2).strip('')
                # print(type(dict[key]))                          # str

    return dict


if __name__ == '__main__':

    file = './data/UK_info.json'
    pattern = r'\|(.*?)\s=\s*(.+)'
    '''|国旗画像 = Flag of the United Kingdom.svg
    |国章画像 = [[ファイル:Royal Coat of Arms of the United Kingdom.svg|85px|イギリスの国章]]
    |国章リンク =（[[イギリスの国章|国章]]）
    ...'''

    basic_info = basic_dict(file, pattern)
    # print(basic_info)　　　　　　　　　　　　　　　　　　　　#　辞書オブジェクト
    for key, value in basic_info.items():
         print(key, ':', value)
