import re
with open("data/UK-text.txt", "r") as f:
    dict1 = {}
    flag_start = False
    key = ""
    for line in f:
        if re.search(r"{{基礎情報\s*国",line):
            flag_start = True
            continue
        if flag_start:
            if re.search(r"^}}$",line):
                break
            rs1 = re.search(r"\|(.*?)\s*=\s*(.*)",line)
            if rs1:
                key = re.sub(r"\'",'',rs1.group(1))
                dict1[key] = re.sub(r"\'",'',rs1.group(2))
            else:
                dict1[key] = dict1[key] + re.sub(r"\'",'',line)
    for key, value in dict1.items():
        print(key,":",value)