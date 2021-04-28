# 03. 円周率
# "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."という文を単語に分解し，
# 各単語の（アルファベットの）文字数を先頭から出現順に並べたリストを作成せよ．


import string

s = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
sw = s.translate(str.maketrans('', '', string.punctuation))  # sw仍是一个字符串，去掉了s中所有的标点符号
l_sw = sw.split()      # 以空格切分字符串，返回切割单元形成的列表
l_alph_n = []
for i in l_sw:
    l_alph_n.append(len(i))
print(l_alph_n)


# 看到另一种使用map(),split()函数的用法