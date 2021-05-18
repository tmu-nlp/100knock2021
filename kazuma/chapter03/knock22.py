import re
from pprint import pprint
with open ("data/US-text.txt", "r") as f:
    for line in f:
        rf = re.findall(r"\[\[Category:(.*?)(?:\|.*)?\]\]",line)
        if rf:
            pprint(rf)
