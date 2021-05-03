path='./knock13.txt'
path1='./col1.txt'
path2='./col2.txt'

with open(path1, mode='r') as file1, open(path2, mode='r') as file2:
	lines1=list(file1)
	lines2=list(file2)
	with open(path, mode='w') as file:
		for line1, line2 in zip(lines1, lines2):
			file.write(line1.replace('\n','\t') + line2)
