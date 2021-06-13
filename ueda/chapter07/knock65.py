with open(r'C:\Git\questions-words-similar.txt', encoding="utf-8") as f:
    total = 0
    crr_cnt = 0
    for line in f:
        line = line.strip().split(" ")
        if line[1].startswith('gram'):
            break
        if len(line) != 6:
            continue
        total += 1
        if line[3] == line[4]:
            crr_cnt+=1
    print("Semantic analogy: {}".format(crr_cnt/total))
    total = 0
    crr_cnt = 0
    for line in f:
        line = line.strip().split(" ")
        if len(line) != 6:
            continue
        total+=1
        if line[3] == line[4]:
            crr_cnt+=1
    print("Syntactic analogy: {}".format(crr_cnt/total))
