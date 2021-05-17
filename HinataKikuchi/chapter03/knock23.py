from eng_json import ans

for lines in ans['text'].split('\n'):
	if lines.find('====') != -1:
		print('section: ' + lines + '\nlevel: ' + str(3))
	elif lines.find('===') != -1:
		print('section: ' + lines + '\nlevel: ' + str(2))
	elif lines.find('==') != -1:
		print('section: ' + lines + '\nlevel: ' + str(1))
