#Receive a natural number $N$ from a command-line argument, and split the input file into $N$ pieces at line boundaries. Confirm the result by using split command.
import sys

file_names = open('/users/kcnco/github/100knock2021/pan/chapter02/popular-names.txt', 'r')
name_lines = [line.replace('\n', '') for line in file_names.readlines()]

print('please type a number')
print('n = ',end = ' ')
n = int(input())

names_n_divided = []
for i in range(n):
    names_n_divided.append(name_lines[i::n])

print(names_n_divided)
