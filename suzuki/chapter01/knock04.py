#“Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.”
#という文を単語に分解し，1, 5, 6, 7, 8, 9, 15, 16, 19番目の単語は先頭の1文字，それ以外の単語は先頭の2文字を取り出し，
#取り出した文字列から単語の位置（先頭から何番目の単語か）への連想配列（辞書型もしくはマップ型）を作成せよ．

Input = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."

text = Input.replace(',', '').replace('.', '')
words = text.split(" ")
first = [1, 5, 6, 7, 8, 9, 15, 16, 19]
ans = {}

for i in range(len(words)):
    if i + 1 in first:
        #print("{} : {}".format(i+1, words[i][0]))
        ans[words[i][0]] = i + 1
    else:
        #print("{} : {}".format(i+1, words[i][:2]))
        ans[words[i][:2]] = i + 1

print(ans)