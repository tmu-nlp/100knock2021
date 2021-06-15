from gensim.models import KeyedVectors

MODEL_PATH = './tmp/GoogleNews-vectors-negative300.bin.gz'
MODEL = KeyedVectors.load_word2vec_format(fname=MODEL_PATH, binary=True)

l = []

with open('./tmp/questions-words.txt', 'r') as f:
    lines = f.readlines()

for line in lines:
    words = line.split()
    if len(words) >= 3:
        w, v = MODEL.most_similar(
            positive=[words[1], words[2]],
            negative=[words[0]],
            topn=1
        )[0]
        l.append(' '.join(words + [w, str(v)]) + '\n')
    else:
        l.append(line)

with open('./tmp/knock64.txt') as f:
    f.writelines(l)
