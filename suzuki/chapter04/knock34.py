#名詞の連接（連続して出現する名詞）を最長一致で抽出せよ．

from knock30 import make_morpheme

f = open('neko.txt.mecab', 'r')
m = make_morpheme(f)
ans = []
second = [] #英語が最長一致では英語が出てきたので一応二番目もみてみる
count = []

for morph in m:
    if morph['pos'] == '名詞':
        count.append(morph['surface'])
        if len(count) > len(ans):
            second = ans
            ans = count
    else:
        count = []

print(' '.join(second))
print(' '.join(ans))