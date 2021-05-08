with open ("data/popular-names.txt","r",encoding = "utf-8") as f1,\
     open ("result/knock11.txt", "w", encoding = "utf-8") as f2 :
    for line in f1:
        f2.write(line.replace("\t"," "))