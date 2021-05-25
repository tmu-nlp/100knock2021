N = int(input('Enter number: '))
with open('c:\Git\popular-names.txt') as f:
    lines = f.readlines()
    l = [(len(lines)+i)//N for i in range(N)]
    for i in range(N):
        with open('c:\Git\knock16_0'+str(i)+'.txt', 'w') as fs:
            fs.writelines("".join(lines[sum(l[:i]):sum(l[:1+i])]))