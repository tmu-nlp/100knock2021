from sklearn.linear_model import LogisticRegression
import pickle
import pandas as pd

train_X, train_y = [], []
with open("./data/train.feature.txt", "r") as f:
    data = f.readlines()
for line in data:
    line = line.strip().split(",")
    train_X.append(list(map(float, line[1:])))
    train_y.append(int(line[0]))

with open("bin/lr.model", "rb") as f:
    lr = pickle.load(f)

def predict(x, lr):
    out = lr.predict_proba(x)
    preds = out.argmax(axis=1)
    probs = out.max(axis=1)
    return preds, probs

preds, probs = predict(train_X, lr)

print(pd.DataFrame([[y, p] for y, p in zip(preds, probs)], columns = ['予測', '確率']).head())
