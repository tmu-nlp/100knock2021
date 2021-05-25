path='./popular-names.txt'
aggregation=set()
with open(path, mode='r') as file:
	lines=list(file)
	for line in lines:
		aggregation.add(line.split('\t')[0])
for val in sorted(aggregation):
	print(val)