from gensim.models import KeyedVectors

MODEL_PATH = './tmp/GoogleNews-vectors-negative300.bin.gz'
MODEL = KeyedVectors.load_word2vec_format(fname=MODEL_PATH, binary=True)

print(MODEL.similarity('United_States', 'U.S.'))
