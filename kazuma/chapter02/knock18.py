# 多重集合だったら、各行に識別番号を一時的に付与して最後に消すと同じ感じ（辞書配列）で行けそう。
with open ("data/popular-names.txt", "r", encoding = "utf-8") as f1,\
     open ("result/knock18.txt", "w", encoding = "utf-8") as f2:
    dict1 = {}
    for line in f1:
        line = line.strip()
        dict1[line] = int(line.split("\t")[2])
    print(len(dict1))
    dict2 = sorted(dict1.items(), key = lambda x:x[1],reverse = True)
    for key, value in dict2:
        f2.write(f"{key}\n")
