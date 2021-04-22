import itertools

input_b = [[0, 1], [2, 3], [4, 5]]
print(list(itertools.chain.from_iterable(input_b)))