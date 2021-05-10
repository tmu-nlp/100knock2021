dict_line = {}

for line in lines:
    dict_line[line[2]] = line

file_4 = open('/users/kcnco/github/100knock2021/pan/chapter02/col4.txt', 'w')
for elem in sorted(dict_line):
    file_4.write('\t'.join(dict_line[elem]) + '\n')

##memo
#a = [5,7,6,3,4,1,2]
#b = sorted(a)
#>>> a
#[5, 7, 6, 3, 4, 1, 2]
#>>> b
#[1, 2, 3, 4, 5, 6, 7]
