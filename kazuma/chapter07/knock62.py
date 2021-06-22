from knock60 import load_vectors
from pprint import pprint
if __name__ == "__main__":
    word_vectors = load_vectors()
    pprint(word_vectors.most_similar("United_States", topn = 10))
