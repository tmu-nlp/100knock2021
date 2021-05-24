with open("popular-names.txt") as f:
    lines = f.readlines()
    sort_by_col3 = sorted(lines, key=lambda line: int(line.split('\t')[2]), reverse=True)
    with open('py/knock18.txt', 'w') as f1:
        for line in sort_by_col3:
            f1.write(line)
