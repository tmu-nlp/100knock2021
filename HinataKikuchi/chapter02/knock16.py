import sys
N=int(sys.argv[1])
path=sys.argv[2]
prefix=sys.argv[3]

with open(path) as file:
	lines=file.readlines()

line_num = round(len(lines) / N)
print(line_num)

for n in range(line_num):
	with open('./' + prefix + str(n) + '.txt', mode='w') as file_n:
		for i in range(N):
			file_n.write(lines[n + i])

###comment###
#splitコマンドがどうやって分割しているかよくわからない。