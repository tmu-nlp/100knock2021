with open('c:\Git\popular-names.txt') as f:
    wordset = set()
    for lines in f:
        wordset.add(lines.split("\t")[0])
    print(wordset)