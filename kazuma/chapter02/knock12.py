with open ("data/popular-names.txt", "r", encoding = "utf-8") as f1,\
     open ("result/col1.txt", "w", encoding = "utf-8") as f2,\
     open ("result/col2.txt", "w", encoding = "utf-8") as f3:
    for line in f1:
        list1 = line.split("\t")
        f2.write(list1[0])
        f2.write("\n")
        f3.write(list1[1])
        f3.write("\n")
