from pprint import pprint
from gensim.models import KeyedVectors

MODEL_PATH = './tmp/GoogleNews-vectors-negative300.bin.gz'
MODEL = KeyedVectors.load_word2vec_format(fname=MODEL_PATH, binary=True)

with open('./combined.csv', 'r') as f:
    lines = f.readlines()[1:]

x = []

for line in lines:
    words = [s.strip() for s in line.split(',')]
    words.append(MODEL.similarity(words[0], words[1]))
    x.append(words)

pprint(x)
