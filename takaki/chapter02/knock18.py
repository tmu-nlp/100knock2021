with open('./popular-names.txt', 'r') as f:
    lines = f.readlines()
    lines = sorted(lines, key=lambda line: int(line.split('\t')[2]), reverse=True)
    for line in lines:
        print(line, end='')
