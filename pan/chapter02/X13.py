#Join the contents of col1.txt and col2.txt, and create a text file whose each line contains the values of the first and second columns (separated by tab character) of the original file. Confirm the result by using paste command.
file_1 = open('/users/kcnco/github/100knock2021/pan/chapter02/col1.txt','r')
col1_lines = [line for line in file_1.readlines()]
file_2 = open('/users/kcnco/github/100knock2021/pan/chapter02/col2.txt','r')
col2_lines = [line for line in file_2.readlines()]

file_3 = open('/users/kcnco/github/100knock2021/pan/chapter02/col3.txt','w')

for i in range(len(col1_lines)):
    file_3.write(col1_lines[i].strip()+'\t'+col2_lines[i])

#memo
#strip() : delete beginning and ending
