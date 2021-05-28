#2つの名詞が「の」で連結されている名詞句を抽出せよ．

from knock30 import make_morpheme

f = open('neko.txt.mecab', 'r')
m = make_morpheme(f)
ans = []

for i in range(len(m)):
    if m[i]['pos'] == '名詞' and i < len(m) - 2: #ひとつ目の名詞を認識（and 以降は out of lange 対策）
        noun1 = m[i]
        if m[i+1]['surface'] == 'の':
            if m[i+2]['pos'] == '名詞': #ふたつ目の名詞を認識
                noun2 = m[i+2]
                ans.append("{}の{}".format(noun1['surface'],noun2['surface']))
                i += 2

f.close()

#確認
print(len(ans))
for i in ans[:5]:
    print(i)

#6045
#彼の掌
#掌の上
#書生の顔
#はずの顔
#顔の真中