##Write a program with the specification:

##Receive a word sequence separated by space
##For each word in the sequence:
##If the word is no longer than four letters, keep the word unchanged
##Otherwise,
##Keep the first and last letters unchanged
##Shuffle other letters in other positions (in the middle of the word)
##Observe the result by giving a sentence, e.g., “I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind “.
##Receive a word sequence separated by space

import random

def random_order(s):
    words = s.split(' ')
    words_random = []
    for word in words:
        if len(word) >= 4:
            temp_word = list(word[1: -1])
            random.shuffle(temp_word)
            word_random = word[0] + ''.join(temp_word) + word[-1]
        else:
            word_random = word
        words_random.append(word_random)
    return ' '.join(words_random)

if __name__ == '__main__':
    s = 'I couldn’t believe that I could actually understand what ' \
        'I was reading : the phenomenal power of the human mind .'
    print(random_order(s))
