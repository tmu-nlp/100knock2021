def n_gram(seq, n):
    char_list, word_list = [], []
    
    #string
    for i in range(len(seq) - n + 1):
        char_list.append(seq[i: i+n])
        
    #list
    seq = seq.split()
    for i in range(len(seq) - n + 1):
        word_list.append(seq[i: i+n])
    
    return char_list, word_list

s = "I am an NLPer"

char_list, word_list = n_gram(s, 2)
print(char_list)
print(word_list)