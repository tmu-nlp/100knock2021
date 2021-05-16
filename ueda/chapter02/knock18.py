with open('c:\Git\popular-names.txt') as f:
    lines = [words.split() for words in f.readlines()]
    lines.sort(key = lambda x: int(x[2]), reverse = True)
    with open("c:\Git\knock18_psort.txt", 'w') as fs:
        fs.writelines('\t'.join(line)+'\n' for line in lines)