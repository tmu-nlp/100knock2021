import gzip
import json
import re
from knock27 import remove_link

def remove_markup():
    dicts = remove_link()
    regex1 = re.compile(r'\{\{(?:lang|仮リンク|Cite web)(?:[^|]*?\|)*?([^|]*?)\}\}', re.MULTILINE)
    regex2 = re.compile(r'\<ref.*\>', re.MULTILINE)
    regex3 = re.compile(r'\<br.*?\>', re.MULTILINE)
    for foo, bar in dicts.items():
        bar = regex1.sub(r'\1', bar)
        bar = regex2.sub('', bar)
        bar = regex3.sub('', bar)
        dicts[foo] = re.sub(r'\{\{.*?\}\}','', bar)
    return dicts 

if __name__ == "__main__":
    dicts = remove_markup()
    for item in dicts.items():
        print(item)
