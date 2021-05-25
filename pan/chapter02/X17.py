#Find distinct strings (a set of strings) of the first column of the file. Confirm the result by using cut, sort, and uniq commands.
names = set()

file_col1 = open('/users/kcnco/github/100knock2021/pan/chapter02/col1.txt', 'r')
col1_lines = [line.strip() for line in file_col1.readlines()]

for line in col1_lines:
    names.add(line)

print(names)

##memo
#set()
#x = set('runoob')
#y = set('google')
#x,y
#(set(['b','r','u','o','n']), set(['e','o','g','l']))
