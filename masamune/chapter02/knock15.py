N = int(input())
n = 0
with open('popular-names.txt') as f:
    for line in f:
        n += 1
    
with open('popular-names.txt') as f:
    i = 0
    for line in f:
        if i >= n - N:
            print(line, end='')
        i += 1

#tail -n 5 popular-names.txt (N = 5)