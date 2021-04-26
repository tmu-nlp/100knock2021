##Implement a function that obtains n-grams from a given sequence object (e.g., string and list). Use this function to obtain word bi-grams and letter bi-grams from the sentence “I am an NLPer”
def n_gram(s, n, char_level=False):
    if char_level:
        inputs = [c for c in s]
    else:
        inputs = s.split(' ')

    return ['|'.join(inputs[i - n: i]) for i in range(n, len(inputs) + 1)]


if __name__ == '__main__':
    s = 'I am an NLPer'
    print('Word level bi-gram:', n_gram(s, 2))
    print('Character level bi-gram:', n_gram(s, 2, char_level=True))
