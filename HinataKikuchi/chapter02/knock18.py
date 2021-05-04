path='./popular-names.txt'
aggregation=set()
with open(path, mode='r') as file:
	lines=list(file)
new_lines=sorted(lines, key=lambda x: x.split('\t')[2])
for val in new_lines:
	print(val.replace('\n',''))