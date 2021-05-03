path='./popular-names.txt'
path1='./col1.txt'
path2='./col2.txt'

with open(path, mode='r') as file:
	lines = list(file)
	with open(path1, mode='w') as file1, open(path2, mode='w') as file2:
		for f in lines:
			file1.write(f.split('\t')[0] + '\n')
			file2.write(f.split('\t')[1] + '\n')
