knockB = [[0, 1], [2, 3], [4, 5]]
output=[]
for value in knockB:
	output += value
print(output)

###ANSWER###
output = sum(knockB, [])
print(output)

import itertools
flatten_list = itertools.chain.from_iterable(knockB)
print(flatten_list)