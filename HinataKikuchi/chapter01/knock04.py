strings = 'Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.'
words = strings.split()
list_idx=[1,5,6,7,8,9,15,16,19]
ans={}
for idx, val in enumerate(words):
	if idx + 1 in list_idx:
		ans[idx]=val[0]
	else:
		ans[idx]=val[:2]
print(ans)

###ANS###
#スライス表記は右側の引数には入らず、一つ小さい整数までになるので注意。
#この場合、elseのvalの配列[:2]は0文字目から1文字目となる。