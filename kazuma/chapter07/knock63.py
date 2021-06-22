from knock60 import load_vectors
from pprint import pprint
if __name__ == "__main__":
    word_vectors = load_vectors()
    word1 = word_vectors["Spain"]
    word2 = word_vectors["Madrid"]
    word3 = word_vectors["Athens"]
    created_word = word1 - word2 + word3
    pprint(word_vectors.most_similar(positive = ["Spain", "Athens"], negative = ["Madrid"], topn = 10))
