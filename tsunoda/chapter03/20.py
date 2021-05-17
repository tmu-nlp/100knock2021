import gzip,json

with gzip.open('./data/jawiki-country.json.gz',mode='rt',encoding='utf=8_sig') as gz_temp:
     jsonlines = gz_temp.read()

lst_country = []
for line in jsonlines.split('\n'):
    if len(line) > 1: # 空行がエラーになるため
      lst_country.append(json.loads(line))

for dct_country in lst_country:
    if dct_country.get('title') == 'イギリス':
        print(dct_country.get('text'))
        break

# {{redirect|UK}}
# {{基礎情報 国
# |略名 = イギリス
# |日本語国名 = グレートブリテン及び北アイルランド連合王国
# |公式国名 = {{lang|en|United Kingdom of Great Britain and Northern Ireland}}<ref>英語以外での正式国名:<br/>
# *{{lang|gd|An Rìoghachd Aonaichte na Breatainn Mhòr agus Eirinn mu Thuath}}（[[スコットランド・ゲール語]]）<br/>
# *{{lang|cy|Teyrnas Gyfunol Prydain Fawr a Gogledd Iwerddon}}（[[ウェールズ語]]）<br/>
# （以下略）