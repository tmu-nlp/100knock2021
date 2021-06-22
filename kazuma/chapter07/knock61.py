from knock60 import load_vectors
import numpy as np

def cos_sim(v1, v2):
    return np.dot(v1, v2) / ( np.sqrt(np.dot(v1,v1)) * np.sqrt(np.dot(v2, v2)) )

if __name__ == "__main__":
    vectors = load_vectors()
    # target_word1 = vectors["United_States"]
    # target_word2 = vectors["U.S."]
    # print(cos_sim(target_word1, target_word2))
    print(vectors.similarity("United_States", "U.S."))