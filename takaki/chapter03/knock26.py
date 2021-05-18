from knock25 import load_england_info
import re
from pprint import pprint

PATTERN = re.compile(r'\'{2,}(.*?)\'{2,}')

def load_england_info2():
    obj = {}
    for key, val in load_england_info().items():
        match = re.search(PATTERN, val)
        if match != None:
            obj[key] = re.search(PATTERN, val).group(1)
        else:
            obj[key] = val
    return obj

if __name__ == '__main__':
    pprint(load_england_info2())
