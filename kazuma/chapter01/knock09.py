import random
sentence = "I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind ."
word_list = sentence.split(" ")
new_sentence = ""
word_array = []
for word in word_list:
    new_word = word
    if len(word) > 4:
        top_chr = word[0]
        last_chr = word[-1]
        midle_chrs = list(word[1:-1])
        random.shuffle(midle_chrs)
        midle_chrs = "".join(midle_chrs)
        new_word = top_chr + midle_chrs + last_chr
    word_array.append(new_word)
new_sentence = " ".join(word_array)
print(new_sentence)
        
