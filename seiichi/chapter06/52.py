from sklearn.linear_model import LogisticRegression
import pickle

train_X, train_y = [], []
with open("./data/train.feature.txt", "r") as f:
    data = f.readlines()
for line in data:
    line = line.strip().split(",")
    train_X.append(list(map(float, line[1:])))
    train_y.append(int(line[0]))

lr = LogisticRegression(max_iter=1000)
lr.fit(train_X, train_y)

with open("bin/lr.model", "wb") as f:
    pickle.dump(lr, f)
