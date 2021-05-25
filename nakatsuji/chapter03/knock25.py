import re, json
from knock20 import text
basic_info = re.findall(r'^\{\{基礎情報.*?$(.*?)^\}\}', text, re.MULTILINE + re.DOTALL)
basic_info = basic_info[0]
pattern = r'^\|(.+?)\s*=\s*(.+?)(?:(?=\n\|)|(?=\n$))'
basic_dic = dict(re.findall(pattern, basic_info, re.MULTILINE + re.DOTALL))

if __name__ == "__main__":
    for k, v in basic_dic.items():
        print(f'{k} : {v}')