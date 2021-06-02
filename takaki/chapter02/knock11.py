with open('popular-names.txt') as f:
    for line in f.readlines():
        print(line.strip().replace('\t', ' '))
