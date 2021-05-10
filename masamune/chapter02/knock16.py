import math
N = int(input())
n = 0
with open('popular-names.txt') as f:
    lines = [line.replace('\n', '').split('\t') for line in f.readlines()]
    n = len(lines)

split_file = []
line_num = math.ceil(n/N)
start = 0
for _ in range(N):
    end = min(start + line_num, n)
    split_file.append(lines[start: end])
    start += line_num

print(*split_file, sep="\n")

#split -l (N分割した時の行数) popular-names.txt