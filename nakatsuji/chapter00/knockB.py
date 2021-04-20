import itertools
in2 = [[0, 1], [2, 3], [4, 5]]
out = list(itertools.chain.from_iterable(in2))
print(out)