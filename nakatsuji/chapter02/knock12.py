with open("popular-names.txt", 'r') as f, open('py/col1.txt', 'w') as f1, open('py/col2.txt', 'w') as f2:
    for line in f.readlines():
        words = line.split('\t')
        f1.write(words[0]+'\n')
        f2.write(words[1]+'\n')
    