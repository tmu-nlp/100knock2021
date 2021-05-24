import gzip
import json
import re
from knock26 import remove_stress

def remove_link():
    dicts = remove_stress()
    #regex = re.compile(r'\[\[(?:[^|]*?\|)*?([^|]*?)\]\]', re.MULTILINE)
    #国歌のところだけ上手くいかない（{{が消える）
    regex1 = re.compile(r'\[\[(?:[^|\[\]]*?\|)?(.*?)\]\]', re.MULTILINE)
    regex2 = re.compile(r'(?:[^|{}]*?\|)?(.*)', re.MULTILINE)
    for foo, bar in dicts.items():
        bar = regex1.sub(r'\1', bar)
        dicts[foo] = regex2.sub(r'\1', bar)
    return dicts 

if __name__ == "__main__":
    dicts = remove_link()
    for item in dicts.items():
        print(item)
