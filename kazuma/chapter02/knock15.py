import sys
n = int(sys.argv[1])
with open ("data/popular-names.txt", "r", encoding = "utf-8") as f1:
    list1 = [i.strip() for i in f1]
    for i in list1[-n:]:
        print(i)