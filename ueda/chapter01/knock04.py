import re

sent = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
sent = re.sub(r"[.,]", "", sent).split()
dicts = {}

for i, word in enumerate(sent):
    if i+1 in [1, 5, 6, 7, 7, 8, 9, 15, 16, 19]:
        dicts[word[:1]] = i+1
    else:
        dicts[word[:2]] = i+1

print(dicts)


    