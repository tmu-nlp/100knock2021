import sys
path='./popular-names.txt'

with open(path) as file:
	lines=file.readlines()
for idx in range(int(sys.argv[1])):
	print(lines[idx].split('\n')[0])