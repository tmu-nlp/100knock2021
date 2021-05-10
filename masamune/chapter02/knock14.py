N = int(input())
with open('popular-names.txt') as f:
    i = 0
    for line in f:
        print(line, end='')
        i += 1
        if i == N:
            break

#head -n 5 popular-names.txt (N = 5)