from gensim.models import KeyedVectors
from pprint import pprint

MODEL_PATH = './tmp/GoogleNews-vectors-negative300.bin.gz'
MODEL = KeyedVectors.load_word2vec_format(fname=MODEL_PATH, binary=True)

pprint(MODEL.most_similar('United_States', topn=10))
