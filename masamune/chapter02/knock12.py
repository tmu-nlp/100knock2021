with open('popular-names.txt') as f:

    with open('col1.txt', 'w') as f1\
        , open('col2.txt', 'w') as f2:
        
        for line in f:
            words = line.split('\t')
            f1.write(words[0]+'\n')
            f2.write(words[1]+'\n')

#cut -f 1 popular-names.txt
#cut -f 2 popular-names.txt
