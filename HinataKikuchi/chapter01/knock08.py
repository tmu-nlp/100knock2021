def cipher(sentence:str):
	if sentence.isupper():
		return
	ans = ''
	for idx, ch in enumerate(sentence):
		if (97 <= ord(ch) and ord(ch) <= 122):
			ans += chr(219 - ord(ch))
		else:
			ans += ch
	return ans

#219 = 'a' + 'z'

print(cipher('Hello?'))
print(cipher(cipher('Hello?')))

###ANS###
#L5の判断を以下のように書き換えたほうがよいかもしれん。
# if (ch.islower()):
#islowerはもちろん、文字列の中身全部判断もできるけど、単発文字もおｋだった。
#Cより便利(((

def ans_cipher(str):
	rep = [chr(219 - ord(x)) if x.islower else x for x in str]
	return ''.join(rep)

print(ans_cipher('Hello?'))
print(ans_cipher(ans_cipher('Hello?')))

#