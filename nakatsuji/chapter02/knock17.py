with open("py/col1.txt") as f:
    lines = f.readlines()

    for i in range(len(lines)):
        lines[i] = lines[i].strip()
    diff_name = set(lines)
    with open("py/knock17.txt", 'w') as f1:
        for name in sorted(diff_name):
            f1.write(name + '\n')