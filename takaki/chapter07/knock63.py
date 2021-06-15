from gensim.models import KeyedVectors
from pprint import pprint

MODEL_PATH = './tmp/GoogleNews-vectors-negative300.bin.gz'
MODEL = KeyedVectors.load_word2vec_format(fname=MODEL_PATH, binary=True)

v = MODEL['Spain'] - MODEL['Madrid'] + MODEL['Athens']
s = MODEL.most_similar(
    positive=['Spain', 'Athens'],
    negative=['Madrid'],
    topn=10
)

pprint(v)
print('\n----------\n')
pprint(s)
