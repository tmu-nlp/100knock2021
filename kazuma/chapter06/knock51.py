from sklearn.feature_extraction.text import CountVectorizer

def create_features_file():
    with open("data/train.txt", "r") as f1,\
         open("data/valid.txt", "r") as f2,\
         open("data/test.txt", "r") as f3,\
         open("data/train.feature.txt", "w") as f4,\
         open("data/valid.feature.txt", "w") as f5,\
         open("data/test.feature.txt", "w") as f6:
        for line in f1:
            line_list = line.strip("\n").split("\t")
            f4. write(f"BOS {line_list[0]} EOS\t{line_list[1]}\n")
        for line in f2:
            line_list = line.strip("\n").split("\t")
            len_title = len(line_list[0].split(" "))
            f5. write(f"BOS {line_list[0]} EOS\t{line_list[1]}\n")
        for line in f3:
            line_list = line.strip("\n").split("\t")
            len_title = len(line_list[0].split(" "))
            f6. write(f"BOS {line_list[0]} EOS\t{line_list[1]}\n")
            
def create_features_file2():
    with open("data/train.txt", "r") as f1,\
         open("data/valid.txt", "r") as f2,\
         open("data/test.txt", "r") as f3,\
         open("data/train.feature.txt", "w") as f4,\
         open("data/valid.feature.txt", "w") as f5,\
         open("data/test.feature.txt", "w") as f6:
        all_sents = []
        for line in f1:
            line_list = line.strip("\n").split("\t")
            all_sents.append(line_list[0])
        for line in f2:
            line_list = line.strip("\n").split("\t")
            all_sents.append(line_list[0])
        for line in f3:
            line_list = line.strip("\n").split("\t")
            all_sents.append(line_list[0])
        vectorizer = CountVectorizer(all_sents, ngram_range = (2,2))
        X = vectorizer.fit_transform(["Be more careful"])
        print(X)
        list1 = X.toarray()
        cnt = 0
        for i in range(int(len(list1)*0.8)):
            cnt += 1
            str1 = ",".join([str(j) for j in list1[i]])
            f4.write(f"{str1}\n")
        print(cnt)
        cnt = 0
        for i in range(int(len(list1)*0.8), int(len(list1)*0.9)):
            cnt += 1
            str1 = ",".join([str(j) for j in list1[i]])
            f5.write(f"{str1}\n")
        print(cnt)
        cnt = 0
        for i in range(int(len(list1)*0.9), len(list1)):
            cnt += 1
            str1 = ",".join([str(j) for j in list1[i]])
            f6.write(f"{str1}\n")
        print(cnt)
if __name__ == "__main__":
    create_features_file()


