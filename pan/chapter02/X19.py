#Find the frequency of a string in the first column, and sort the strings by descending order of their frequencies. Confirm the result by using cut, uniq, and sort commands.
file = open('/users/kcnco/github/100knock2021/pan/chapter02/popular-names.txt', 'r')
lines = [line.strip() for line in file.readlines()]
dict_line = {}

for line in lines:
    if line[0] not in dict_line:
        dict_line[line[0]] = 1
    else:
        dict_line[line[0]] = dict_line[line[0]] + 1

print(sorted(dict_line.items(), key=lambda x: x[1], reverse=True))

##memo
#dict.items()
