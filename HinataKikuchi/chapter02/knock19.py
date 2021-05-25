path='./popular-names.txt'
aggregation=set()
with open(path, mode='r') as file:
	lines=list(file)
	for line in lines:
		aggregation.add(line.split('\t')[0])

new_line = [line.split('\t')[0] for line in lines]

l = []
for set_val in aggregation:
	key_n = new_line.count(set_val)
	l.append([key_n,set_val])
ans = sorted(l, reverse=True)
for val in ans:
	print(val[0],val[1])

###ANS###
#default dictもあるよ！でもカウンタのがより完結！