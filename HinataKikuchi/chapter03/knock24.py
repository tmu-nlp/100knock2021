from eng_json import ans

files = []
for lines in ans['text'].split('\n'):
	if lines.find('[[ファイル') != -1:
		files.append(lines[7:].split('|')[0])
media_files = []
for file in files:
	if file.find('[[ファイル') != -1:
		media_files.append(file.replace(' [[ファイル:', ''))
	elif file.find('jpg')!=-1 or file.find('PNG')!=-1 or file.find('JPG')!=-1 or file.find('svg')!=-1:
		media_files.append(file)

print(media_files)