#Count the number of lines of the file. Confirm the result by using wc command.
file = open('/users/kcnco/github/100knock2021/pan/chapter02/popular-names.txt','r')
lines = [line for line in file.readlines()]
print('The number is', end = ' ')
print(len(lines))
#The number is 2780#

##memo
#file.readlines() : read all the lines
#file.readline() : read one line
#file.read() : read words

#str = 'moon'
#len(str)
#4
#s = [1,2,3]
#len(s)
#3
