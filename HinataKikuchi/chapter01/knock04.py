strings = 'Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.'
words = strings.split()
list_idx=[1,5,6,7,8,9,15,16,19]
ans={}
for idx, val in enumerate(words):
	if idx + 1 in list_idx:
		ans[idx]=val[0]
	else:
		ans[idx]=val[1]
print(ans)