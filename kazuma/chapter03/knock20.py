import json
with open ("data/jawiki-country.json", "r") as f,\
     open ("data/UK-text.txt", "w") as f2:
    for line in f:
        dict1 = json.loads(line)
        if dict1["title"] == "イギリス":
            f2.write(dict1["text"])
