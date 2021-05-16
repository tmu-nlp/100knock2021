with open ("result/col1.txt", "r", encoding = "utf-8") as f1,\
     open ("result/col2.txt", "r", encoding = "utf-8") as f2,\
     open ("result/knock13.txt", "w", encoding = "utf-8") as f3:

    # list1 = []
    # list2 = []
    # for line in f1:
    #     list1.append(line.strip())
    # for line in f2:
    #     list2.append(line.strip())

    list1 = [i.strip() for i in f1]
    list2 = [i.strip() for i in f2]

    # for i in range(len(list1)):
    #     f3.write(list1[i])
    #     f3.write("\t")
    #     f3.write(list2[i])
    #     f3.write("\n")

    for i in range(len(list1)):
        f3.write(f"{list1[i]}\t{list2[i]}\n")
