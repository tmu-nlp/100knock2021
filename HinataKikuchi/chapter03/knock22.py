from eng_json import ans

categories = []
for text in ans['text'].split('\n'):
	if text.find('[[Category:') != -1:
		categories.append(text[11:].strip(']]'))
		print(text[11:].strip(']]'))
