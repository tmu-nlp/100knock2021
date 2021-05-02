path = './popular-names.txt'

with open(path) as file:
	buf = file.read()
	print(buf.replace('\t',' '))
