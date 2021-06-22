from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

# def get_features_and_category():
#     with open("data/train.feature.txt", "r") as f1,\
#          open("data/valid.feature.txt", "r") as f2,\
#          open("data/test.feature.txt", "r") as f3:
#         all_sents = []
#         all_categs = []
#         for line in f1:
#             line_list = line.strip("\n").split("\t")
#             all_sents.append(line_list[0])
#             all_categs.append(line_list[1])
#         for line in f2:
#             line_list = line.strip("\n").split("\t")
#             all_sents.append(line_list[0])
#             all_categs.append(line_list[1])
#         for line in f3:
#             line_list = line.strip("\n").split("\t")
#             all_sents.append(line_list[0])
#             all_categs.append(line_list[1])
#         vectorizer = CountVectorizer(ngram_range= (2,2))
#         X = vectorizer.fit_transform(all_sents)
#         print()
        
#         train, valid, test = (X[:int(len(all_categs)*0.8)], all_categs[:int(len(all_categs)*0.8)]), (X[int(len(all_categs)*0.8):int(len(all_categs)*0.9)],all_categs[int(len(all_categs)*0.8):int(len(all_categs)*0.9)]), (X[int(len(all_categs)*0.9):],all_categs[int(len(all_categs)*0.9):])
#         return vectorizer, train, valid, test

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

def train():
    with open("data/train.feature.txt", "r") as f1:
        vectorizer, train, valid, test= get_features_and_category()
        lr = LogisticRegression(penalty = "none") 
        lr.fit(train[0], train[1])
        return lr, (train, valid, test), vectorizer

if __name__ == "__main__":
    lr, data, vectorizer= train()
    print(lr.score(data[0][0], data[0][1]))