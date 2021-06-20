from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz', binary=True)

with open('./questions-words.txt') as i_f, open('./questions-words-re.txt', 'w') as o_f:
    for line in i_f:
        if line[0] == ":":
            k, category = line.split()
            o_f.write(category+'\n')
            continue

        line = line.strip().split()
        res = model.most_similar(positive=[line[1], line[2]], negative=[line[0]], topn=1)[0]
        o_f.write(' '.join(line + [res[0], str(res[1]), '\n']))