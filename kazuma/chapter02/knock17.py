with open ("data/popular-names.txt", "r", encoding = "utf-8") as f1:
    col1_set = set()
    for line in f1:
        col1_set.add(line.split("\t")[0].strip())
    print(len(col1_set))
