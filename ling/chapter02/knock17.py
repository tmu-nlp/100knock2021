with open ("popular-names.txt", "r", encoding = "utf-8") as f:
    col1 = set()
    for line in f:
        col1.add(line.split("\t")[0].strip())
    print(len(col1))