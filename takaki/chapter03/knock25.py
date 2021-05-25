from knock20 import load_england
import re
from pprint import pprint

PATTERN1 = r'^\{\{基礎情報.*?$(.*?)^\}\}'
PATTERN2 = r'^\|(.+?)\s*=\s*(.+?)(?:(?=\n\|)|(?=\n$))'

def load_england_info():
    template = re.search(PATTERN1, load_england(), re.MULTILINE + re.DOTALL)
    return dict(re.findall(PATTERN2, template.group(1), re.MULTILINE + re.DOTALL))

if __name__ == '__main__':
    pprint(load_england_info())
