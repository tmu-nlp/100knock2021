import gzip
import json
import re
from knock20 import load_json

def get_info():
    regex1= re.compile(r'^\{\{基礎情報.*?$(.*?)^\}\}$', re.MULTILINE+re.DOTALL)
    info = regex1.findall(load_json())
    regex2= re.compile(r'^\|(.*?)\s*\=\s*(.*?)$', re.MULTILINE)
    dicts={}
    for info_cat in regex2.findall(info[0]):
        dicts[info_cat[0]] = info_cat[1]
    return dicts

if __name__ == "__main__":
    dicts = get_info()
    for item in dicts.items():
        print(item)
