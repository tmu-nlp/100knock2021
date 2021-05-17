import json,re

with open('./jawiki-country.json','r',encoding='utf-8') as f:
    for line in f:
        line=json.loads(line)
        if line['title']=='イギリス':
            text=line['text']
            break

pattern='^(.*\[\[Category:.*\]\].*)$'
result='\n'.join(re.findall(pattern,text,re.MULTILINE))
print(result)

'''
[.] 改行以外の任意の文字
[*] 直前の正規表現を 0 回以上、できるだけ多く繰り返したもの
[\] エスケープ文字
[()] 丸括弧で囲まれた正規表現にマッチするとともに、グループの開始と終了を表す
[$] 文字列の末尾、あるいは文字列の末尾の改行の直前にマッチ
'''