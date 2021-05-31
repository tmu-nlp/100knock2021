from knock30 import load_mecab
from collections import Counter

def word_counter(num):
    word_count = Counter()
    for line in load_mecab():
        for morpheme in line:
            word_count[morpheme['surface']]+=1
    return word_count.most_common(num)

if __name__ == "__main__":
    print(word_counter(None))