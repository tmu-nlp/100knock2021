#与えられた文字列の各文字を，以下の仕様で変換する関数cipherを実装せよ．

#英小文字ならば(219 - 文字コード)の文字に置換
#その他の文字はそのまま出力
#この関数を用い，英語のメッセージを暗号化・復号化せよ．

def cipher(text):
    ans = []

    for l in text:
        if ord("a") <= ord(l) <= ord("z"): #chがaからzの文字コードの範囲内にあるかどうか判定
            ans.append(chr(219 - ord(l)))
        else: ans.append(l)
    
    return ''.join(ans) #文字列 = '区切り文字'.join(リスト)

print(cipher("Hello World!"))
print(cipher(cipher("Hello World!"))) #219 = 97('a'の文字コード) + 122('z'の文字コード)　なので複合も同じ関数でできる
