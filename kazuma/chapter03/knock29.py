import re
import urllib.parse, urllib.request
import json

def rm_internal_link(snt):
    return re.sub(r"\[\[.*?\|(.*)?\]\]",r"\1",snt)

def rm_emp_mark(snt):
    return re.sub(r"\'",'',snt)
    
def remover_knock27(snt):
    if re.match(r"\[\[ファイル",snt):return snt
    else:return rm_internal_link(rm_emp_mark(snt))

def remover_knock28(snt):
    return re.sub(r"<.*>",'',remover_knock27(snt))

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
                key = remover_knock28(rs1.group(1))
                dict1[key] = remover_knock28(rs1.group(2))
            else:
                dict1[key] = dict1[key] + remover_knock28(line)
    
    flag_name_f = dict1["国旗画像"]
    url = 'https://www.mediawiki.org/w/api.php?' \
    + 'action=query' \
    + '&titles=File:' + urllib.parse.quote(flag_name_f) \
    + '&format=json' \
    + '&prop=imageinfo' \
    + '&iiprop=url'

    request = urllib.request.Request(url, headers={'User-Agent': 'kz'})
    connection = urllib.request.urlopen(request)
    data = json.loads(connection.read().decode())
    url = data['query']['pages'].popitem()[1]['imageinfo'][0]['url']
    print(url)