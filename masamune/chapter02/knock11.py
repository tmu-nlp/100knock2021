with open("popular-names.txt") as f:
    for line in f:
        line = line.replace('\t', ' ')
        print(line, end='')        

#expand -t 1 popular-names.txt