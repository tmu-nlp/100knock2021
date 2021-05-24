import re

with open('uk.json') as file:
    for line in file:
        line = line.replace('\n', '')
        if re.match(r'\[\[Category:.*\]\]', line):
            #文字列抽出
            print(''.join(re.findall(r'\[\[Category:(.*?)(?:\|.*)?\]\]', line)))
            
            #(.*?)・・・()グループ化
            #(?:\|.*)・・・(?:)グループ化しない、|以降は抽出しなくなる
            #?・・・直前の正規表現を0or1回繰り返したものにマッチさせる