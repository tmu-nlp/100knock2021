with open('c:\Git\popular-names.txt') as f, open('c:\Git\knock11-p.txt', 'w') as fp:
    fp.writelines(lines.replace("\t", " ") for lines in f.readlines())