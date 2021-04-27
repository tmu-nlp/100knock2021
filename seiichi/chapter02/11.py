with open('./dat/popular-names.txt') as f, open('./out_py/11.txt', 'w') as f2:
    for line in f.readlines():
        f2.write(line.replace("\t", " "))
