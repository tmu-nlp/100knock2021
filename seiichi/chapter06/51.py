import re
import spacy
import nltk
from collections import Counter
from sklearn.feature_extraction.text import TfidfTransformer

def load(path):
    ret = []
    with open(path, "r") as f:
        data = f.readlines()
        for line in data:
            line = line.strip().split("\t")
            ret.append(line)
    return ret

def save(x, y, path):
    tmp = "\n".join([str(label)+","+",".join(list(map(str, sent))) for sent, label in zip(x, y)])
    with open(path, "w") as f:
        f.write(tmp)

nlp = spacy.load('en')
stemmer = nltk.stem.snowball.SnowballStemmer(language='english')

def tokenize(x):
    x = re.sub(r'\s+', ' ', x)
    x = nlp.make_doc(x)
    x = [stemmer.stem(doc.lemma_.lower()) for doc in x]
    return x

train, valid, test = load("./data/train.txt"), load("./data/valid.txt"), load("./data/test.txt")
ttrain, tvalid, ttest = [[cat, tokenize(line)] for cat, line in train], [[cat, tokenize(line)] for cat, line in valid], [[cat, tokenize(line)] for cat, line in test]

counter = Counter([token for _, tokens in ttrain for token in tokens])
vocab = [token for token, freq in counter.most_common() if 2 < freq < 300]
word2id = {w:i for i, w in enumerate(vocab)}

with open("./data/vocab.txt", "w") as f:
    f.write("\n".join(vocab))

def unigram_bow(sent):
    l = [0 for _ in range(len(vocab))]
    for w in sent:
        if w in vocab:
            l[word2id[w]] += 1
    return l

categories = ['b', 't', 'e', 'm']

tfidf = TfidfTransformer(smooth_idf=False)

train_X, train_y = [unigram_bow(s) for _, s in ttrain], [categories.index(label) for label, _ in ttrain]
valid_X, valid_y = [unigram_bow(s) for _, s in tvalid], [categories.index(label) for label, _ in tvalid]
test_X, test_y = [unigram_bow(s) for _, s in ttest], [categories.index(label) for label, _ in ttest]

model = tfidf.fit(train_X)
train_X, valid_X, test_X = model.transform(train_X).toarray(), model.transform(valid_X).toarray(), model.transform(test_X).toarray()

save(train_X, train_y, "./data/train.feature.txt")
save(valid_X, valid_y, "./data/valid.feature.txt")
save(test_X, test_y, "./data/test.feature.txt")

