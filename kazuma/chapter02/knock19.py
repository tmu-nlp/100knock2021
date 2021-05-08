from collections import defaultdict
with open ("data/popular-names.txt", "r", encoding = "utf-8") as f1,\
     open ("result/knock19.txt", "w", encoding = "utf-8") as f2:
    df_dict1 = defaultdict(lambda:0)
    for line in f1:
        df_dict1[line.split("\t")[0].strip()] += 1
    dict1 = dict(df_dict1)
    dict2 = sorted(dict1.items(), key = lambda x:x[1],reverse = True)
    for key,value in dict2:
        f2.write(f"{value} {key}\n")