import sys
path='./popular-names.txt'

with open(path) as file:
	lines=file.readlines()
for idx in range(int(sys.argv[1])):
	print(lines[idx].split('\n')[0])

###ANS###
# N以上の行を読みだしてる。
# file.readlines()[:N]とかしたほうがメモリ節約になる。