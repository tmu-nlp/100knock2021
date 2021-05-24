import re
with open("data/UK-text.txt", "r") as f:
    for line in f:
        rs = re.search(r"\[\[Category:.*\]\]",line)
        if rs:
            print(rs.group())