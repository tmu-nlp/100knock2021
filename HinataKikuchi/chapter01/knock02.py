str1='パトカー'
str2='タクシー'
str3 = 'リムジンバス'
ans=[]
for val in zip(str1, str2):
	# print(val)
	ans+=val
print(ans)

###ANS###
print(''.join([word1 + word2 for word1, word2 in zip(str1,str2)]))
#長さが違うときで、ない文字を全角空白で埋めたい場合は？
#-> from itertools import zip_longestを使うとよいよ！
#word数が任意で任意のstringを1文字目から連結する場合は？

from itertools import zip_longest
print(''.join([word1 + word2 + word3 for word1, word2,word3 in zip_longest(str1,str2,str3)]))
