def create_ngram(n, sequence):
    result_list = []
    point_index = 0
    for i in range(0, len(sequence)-n+1):
        result_list.append(sequence[i:i+n])
    return result_list

n = 2
target_sentence = "I am an NLPer"
word_bigram_list = create_ngram(n,target_sentence.split(" "))
print(word_bigram_list)
character_bigram_list = create_ngram(n,target_sentence)
print(character_bigram_list)