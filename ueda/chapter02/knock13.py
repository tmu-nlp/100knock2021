with open('c:\Git\col1_2.txt', 'w') as f, open('c:\Git\col1.txt') as col1, open('c:\Git\col2.txt') as col2:
    f.writelines(word1.strip()+'\t'+word2 for word1, word2 in zip(col1.readlines(), col2.readlines()))
        