with open('col1.txt') as f1\
    , open('col2.txt') as f2:

    with open('merge_col.txt', 'w') as f:
        for line1, line2 in zip(f1, f2):
            line1 = line1.replace('\n', '')
            line2 = line2.replace('\n', '')
            f.write(f"{line1}\t{line2}\n")

#paste -d '\t' col1.txt col2.txt