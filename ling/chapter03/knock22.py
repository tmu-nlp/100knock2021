import json,re

with open('./jawiki-country.json','r',encoding='utf-8') as f:
    for line in f:
        line=json.loads(line)
        if line['title']=='イギリス':
            text=line['text']
            break

pattern='^.*\[\[Category:(.*?)(?:\|.*)?\]\].*$'
result='\n'.join(re.findall(pattern,text,re.MULTILINE))
print(result)
'''
? 直前の正規表現を 0 回か 1 回繰り返したものにマッチさせる結果の正規表現

(?:...)
普通の丸括弧の、キャプチャしない版です。丸括弧で囲まれた正規表現にマッチしますが、
このグループがマッチした部分文字列は、マッチを実行したあとで回収することも、
そのパターン中で以降参照することもできません 
'''