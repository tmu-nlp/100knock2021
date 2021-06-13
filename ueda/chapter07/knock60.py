import gzip
import gensim

def struct_vector():
    model = gensim.models.KeyedVectors.load_word2vec_format(r'C:\Git\GoogleNews-vectors-negative300.bin.gz', binary=True)
    return model

if __name__ == "__main__":
    print(struct_vector()["United_States"])
