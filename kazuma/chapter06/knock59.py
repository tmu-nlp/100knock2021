from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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

def multi_svm():
    with open("data/train.feature.txt", "r") as f1:
        vectorizer, train, valid, test= get_features_and_category()
        C = 1.
        kernel = 'rbf'
        gamma  = 0.01
        estimator = SVC(C=C, kernel=kernel, gamma=gamma)
        classifier = OneVsRestClassifier(estimator)
        classifier.fit(train[0], train[1])
        pred_y = classifier.predict(test[0])
        print (f'One-versus-the-rest: {accuracy_score(test[1], pred_y)}')
        classifier2 = SVC(C=C, kernel=kernel, gamma=gamma)
        classifier2.fit(train[0], train[1])
        pred_y2 = classifier2.predict(test[0])
        print (f'One-versus-one: {accuracy_score(test[1], pred_y2)}')

def logistic_regression():
    with open("data/train.feature.txt", "r") as f1:
        vectorizer, train, valid, test= get_features_and_category()
        lr = LogisticRegression(C = 10) 
        lr.fit(train[0], train[1])
        print("logistaic regresssion:", lr.score(test[0], test[1]))
        
def knock59():
    # multi_svm()
    logistic_regression()

if __name__ == "__main__":
    knock59()
    