from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

def get_features_and_category():
    with open("data/train.txt", "r") as f1,\
         open("data/valid.txt", "r") as f2,\
         open("data/test.txt", "r") as f3:
        all_sents = []
        all_categs = []
        for line in f1:
            line_list = line.strip("\n").split("\t")
            all_sents.append(line_list[0])
            all_categs.append(line_list[1])
        for line in f2:
            line_list = line.strip("\n").split("\t")
            all_sents.append(line_list[0])
            all_categs.append(line_list[1])
        for line in f3:
            line_list = line.strip("\n").split("\t")
            all_sents.append(line_list[0])
            all_categs.append(line_list[1])
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(all_sents)
        print()
        
        train, valid, test = (X[:int(len(all_categs)*0.8)], all_categs[:int(len(all_categs)*0.8)]), (X[int(len(all_categs)*0.8):int(len(all_categs)*0.9)],all_categs[int(len(all_categs)*0.8):int(len(all_categs)*0.9)]), (X[int(len(all_categs)*0.9):],all_categs[int(len(all_categs)*0.9):])
        return vectorizer, train, valid, test

def train(constraint):
    with open("data/train.feature.txt", "r") as f1:
        vectorizer, train, valid, test= get_features_and_category()
        lr = LogisticRegression(penalty = "l2",C = constraint)
        lr.fit(train[0], train[1])
        return lr, (train, valid, test), vectorizer

def knock58():
    constraints = [0.01, 0.1, 1, 10, 100]
    train_acc, valid_acc, test_acc = [], [], []
    for constraint in constraints:
        lr, data, vectorizer = train(constraint)
        train_acc.append(lr.score(data[0][0], data[0][1]))
        valid_acc.append(lr.score(data[1][0], data[1][1]))
        test_acc.append(lr.score(data[2][0], data[2][1]))
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.xscale("log")

    ax.plot(constraints, train_acc, label='train')
    ax.plot(constraints, valid_acc, label='valid')
    ax.plot(constraints, test_acc, label='test')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    knock58()
    