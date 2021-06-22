from gensim.models import KeyedVectors

def create_vectors():
    return KeyedVectors.load_word2vec_format("data/GoogleNews-vectors-negative300.bin", binary=True)

def save_vectors(word_vectors):
    word_vectors.save("vectors.kv")

def load_vectors():
    return KeyedVectors.load("vectors.kv")

if __name__ == "__main__":
    word_vectors = create_vectors()
    save_vectors(word_vectors)
    print(word_vectors["United_States"])


    
