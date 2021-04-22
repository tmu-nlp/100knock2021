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

print(cipher('Hello?'))
