from gensim.models import KeyedVectors

MODEL_PATH = './tmp/GoogleNews-vectors-negative300.bin.gz'
MODEL = KeyedVectors.load_word2vec_format(fname=MODEL_PATH, binary=True)

with open('./tmp/questions-words.txt', 'r') as f:
    lines = f.readlines()

result = []
category = ''

for line in lines:
    words = line.split()
    if words[0] == ':':
        category = words[1]
    else:
        word, cos = MODEL.most_similar(
            positive=[words[1], words[2]],
            negative=[words[0]],
            topn=1
        )[0]
        result.append(
            ' '.join([category] + words + [word, str(cos)]) + '\n'
        )

with open('./tmp/knock64.txt') as f:
    f.writelines(result)
