with open('c:\Git\popular-names.txt') as f, open('c:\Git\col1.txt', 'w') as col1, open('c:\Git\col2.txt', 'w') as col2:
    for lines in f:
        words = lines.split("\t")
        col1.write(words[0]+'\n'), col2.write(words[1]+'\n')