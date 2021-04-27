# 04. 元素記号
# "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
# という文を単語に分解し，1, 5, 6, 7, 8, 9, 15, 16, 19番目の単語は先頭の1文字，それ以外の単語は先頭に2文字を取り出し，
# 取り出した文字列から単語の位置（先頭から何番目の単語か）への連想配列（辞書型もしくはマップ型）を作成せよ．


# 字符串切分为列表，存放指定位置信息
txt = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
txt = txt.replace('.', '').split()
id_list = [1, 5, 6, 7, 8, 9, 15, 16, 19]

# 枚举字符串列表元素，当属于指定位置时，截取元素首字母；否则，截取元素头两个字母，结果返回字典型。
result = {}
for i, word in enumerate(txt):
    if i + 1 in id_list:
        result[i + 1] = word[0]
    else:
        result[i + 1] = word[:2]

print(result)