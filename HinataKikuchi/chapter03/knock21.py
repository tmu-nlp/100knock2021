from eng_json import ans

# print(type(ans))

# print(ans['text'].split('\n'))

list_ans =[]
for i in ans['text'].split('\n'):
	if i.find('Category:') != -1:
		list_ans.append(i)
		print(i)