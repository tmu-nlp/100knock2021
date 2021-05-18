import sys
n = int(sys.argv[1])
with open('popular-names.txt') as f, open('py/knock15.txt', 'w') as f1:
    lines = f.readlines()
    for i in range(n):
        f1.write(lines[-(n-1-i+1)])