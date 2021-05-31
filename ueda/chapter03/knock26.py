import gzip
import json
import re
from knock25 import get_info

def remove_stress():
    dicts = get_info()
    regex = re.compile(r'\'{2,5}')
    for foo, bar in dicts.items():
        dicts[foo] = regex.sub('', bar)
    return dicts 

if __name__ == "__main__":
    dicts = remove_stress()
    for item in dicts.items():
        print(item)
