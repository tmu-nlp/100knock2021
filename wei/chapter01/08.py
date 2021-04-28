# 08. 暗号文
# 与えられた文字列の各文字を，以下の仕様で変換する関数cipherを実装せよ．
# 英小文字ならば(219 - 文字コード)の文字に置換
# その他の文字はそのまま出力
# この関数を用い，英語のメッセージを暗号化・復号化せよ．


# 使用ord()函数，返回单个字符的ascii值（0-255）.chr()函数：输入0-255的一个整数，返回其对应的ascii符号。

def cipher(txt):
    n_txt = ''
    for i in range(len(txt)):
        if ord(txt[i]) in range(ord('a'), ord('{')):
            n_chr = chr(219 - ord(txt[i]))
        else:
            n_chr = txt[i]
        n_txt += n_chr

    return n_txt


if __name__ == '__main__':
    txt = 'Hello World! I love NLP.ハローワールド！'
    print(cipher(txt))
