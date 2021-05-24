import sys
n = int(sys.argv[1])
with open('popular-names.txt') as f, open('py/knock14.txt', 'w') as f1:
    for i in range(n):
        f1.write(f.readline())