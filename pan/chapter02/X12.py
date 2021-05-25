#Extract the value of the first column of each line, and store the output into col1.txt. Extract the value of the second column of each line, and store the output into col2.txt. Confirm the result by using cut command.
file = open('/users/kcnco/github/100knock2021/pan/chapter02/popular-names.txt','r')
lines = [line for line in file.readlines()]

file_1 = open('/users/kcnco/github/100knock2021/pan/chapter02/col1.txt','w')
file_2 = open('/users/kcnco/github/100knock2021/pan/chapter02/col2.txt','w')

for line in lines:
    line = line.split()
    file_1.write(line[0]+'\n')
    file_2.write(line[1]+'\n')
