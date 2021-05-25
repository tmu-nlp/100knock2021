from knock26 import load_england_info2
import re
from pprint import pprint

PATTERN1 = re.compile(r'\[{2}.*?\|(.*?)\]{2}')
PATTERN2 = re.compile(r'\[{2}(.*?)\]{2}')

def load_england_info3():
    obj = {}
    for key, val in load_england_info2().items():
        match1 = re.search(PATTERN1, val)
        if match1 != None:
            obj[key] = match1.group(1)
        else:
            match2 = re.search(PATTERN2, val)
            if match2 != None:
                obj[key] = match2.group(1)
            else:
                obj[key] = val
    return obj

if __name__ == '__main__':
    pprint(load_england_info3())
