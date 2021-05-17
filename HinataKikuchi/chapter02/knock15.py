import sys
path='./popular-names.txt'

with open(path) as file:
	lines=file.readlines()
for idx in reversed(range(int(sys.argv[1]))):
	print(lines[(len(lines)-1) - idx].split('\n')[0])

###ANS###
# 本当はN行だけでいいよね…
# dequeって関数を使うとtailと同じように出せますよ。
# (https://docs.python.org/ja/3/library/collections.html#deque-recipes)