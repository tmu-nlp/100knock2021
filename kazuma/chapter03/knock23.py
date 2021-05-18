import re
with open("data/US-text.txt") as f:
    for line in f:
        rf = re.search(r".*?(={2,})\s*(.+?)\s*\1.*",line)
        if rf:
            print(rf.group(2),":",len(rf.group(1))-1)