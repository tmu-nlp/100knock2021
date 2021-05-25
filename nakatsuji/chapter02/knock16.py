import sys
n = int(sys.argv[1])
with open("popular-names.txt") as f:
    lines = f.readlines()
    slice_num = len(lines) // n
    for i in range(n):
        file_name = "py/knock16-{}".format(i)
        with open(file_name, 'w') as f1:
            start = i * slice_num
            if i == n-1:
                f1.write("".join(lines[start:]))
            else:
                f1.write("".join(lines[start:start+slice_num]))