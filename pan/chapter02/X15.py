#Receive a natural number $N$ from a command-line argument, and output the last $N$ lines of the file. Confirm the result by using tail command.
import sys

filename = open('/users/kcnco/github/100knock2021/pan/chapter02/popular-names.txt', 'r')
print('please input a number')
print('n = ',end = ' ')
n = int(input())
for line in filename.readlines()[-n:]:
    print(line[:-1])
