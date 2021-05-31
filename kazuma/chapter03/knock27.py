import re
def rm_internal_link(snt):
    return re.sub(r"\[\[.*?\|(.*)?\]\]",r"\1",snt)

def rm_emp_mark(snt):
    return re.sub(r"\'",'',snt)

def remover_knock27(snt):
    if re.match(r"\[\[ファイル",snt):return snt
    else:return rm_internal_link(rm_emp_mark(snt))

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
                key = remover_knock27(rs1.group(1))
                dict1[key] = remover_knock27(rs1.group(2))
            else:
                dict1[key] = dict1[key] + remover_knock27(line)
    for key, value in dict1.items():
        print(key,":",value)