path='./popular-names.txt'
path1='./col1.txt'
path2='./col2.txt'

with open(path, mode='r') as file:
	lines = list(file)
	print(lines[1].split('\t')[1])
	# with open(path1, mode='w') as file1, open(path2, mode='w') as file2:
	# 	for f1 in [line for idx, line in enumerate(lines) if idx%2==0]:
	# 		file1.write(f1)
	# 	for f2 in [line for idx, line in enumerate(lines) if idx%2==1]:
	# 		file2.write(f2)
